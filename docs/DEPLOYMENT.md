# WhispRX Production Deployment Guide

Complete guide for deploying WhispRX to production on Kubernetes with GPU support.

---

## Prerequisites

### Infrastructure
- **Kubernetes cluster** 1.25+
- **NVIDIA GPU nodes** with CUDA 12.1+
- **NVIDIA Device Plugin** for Kubernetes
- **Persistent storage** (50GB+ per instance)
- **LoadBalancer** or Ingress controller

### Tools
- kubectl
- helm 3
- docker (for building images)

---

## Quick Start

```bash
# 1. Build and push Docker image
docker build -f services/whisprx_api/Dockerfile -t your-registry/whisprx:latest .
docker push your-registry/whisprx:latest

# 2. Install with Helm
helm install whisprx charts/whisprx/ \
    --set image.repository=your-registry/whisprx \
    --set image.tag=latest

# 3. Check status
kubectl get pods -l app=whisprx
kubectl logs -f deployment/whisprx
```

---

## Step-by-Step Deployment

### 1. Setup NVIDIA Device Plugin

If not already installed:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

Verify GPUs are available:

```bash
kubectl get nodes -o json | jq '.items[].status.capacity'
# Should show "nvidia.com/gpu": "1" (or more)
```

### 2. Create Namespace

```bash
kubectl create namespace whisprx-prod
kubectl label namespace whisprx-prod monitoring=enabled
```

### 3. Create Persistent Volume for Models

```yaml
# pv-models.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: whisprx-models
  namespace: whisprx-prod
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd  # Use your storage class
```

Apply:

```bash
kubectl apply -f pv-models.yaml
```

### 4. Download Models to PVC

Create a job to download models:

```yaml
# job-download-models.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: download-models
  namespace: whisprx-prod
spec:
  template:
    spec:
      containers:
      - name: downloader
        image: python:3.11
        command:
          - /bin/bash
          - -c
          - |
            pip install huggingface_hub
            mkdir -p /models/whisper /models/llm /models/tts
            # Download models here
            python -c "from huggingface_hub import snapshot_download; \
                       snapshot_download('microsoft/Phi-3-mini-4k-instruct', local_dir='/models/llm/phi-3')"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: whisprx-models
      restartPolicy: Never
```

Apply and wait:

```bash
kubectl apply -f job-download-models.yaml
kubectl wait --for=condition=complete job/download-models -n whisprx-prod --timeout=30m
```

### 5. Configure Helm Values

Create `production-values.yaml`:

```yaml
image:
  repository: your-registry/whisprx
  tag: "0.2.0"  # Use specific version
  pullPolicy: Always

replicaCount: 2  # For high availability

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 16Gi
    cpu: 4000m
  requests:
    nvidia.com/gpu: 1
    memory: 12Gi
    cpu: 2000m

persistence:
  enabled: true
  existingClaim: whisprx-models  # Use pre-created PVC
  mountPath: /models

env:
  - name: WHISPER_MODEL_PATH
    value: "base"
  - name: LLM_MODEL_NAME
    value: "/models/llm/phi-3"  # Local path
  - name: TTS_MODEL_PATH
    value: "/models/tts/piper/model.onnx"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/websocket-services: whisprx
  hosts:
    - host: whisprx.yourcompany.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: whisprx-tls
      hosts:
        - whisprx.yourcompany.com

serviceMonitor:
  enabled: true
  interval: 15s
```

### 6. Deploy with Helm

```bash
helm install whisprx charts/whisprx/ \
    -f production-values.yaml \
    -n whisprx-prod
```

### 7. Verify Deployment

```bash
# Check pods are running
kubectl get pods -n whisprx-prod

# Check GPU allocation
kubectl describe pod -n whisprx-prod -l app=whisprx | grep nvidia.com/gpu

# Check logs
kubectl logs -f -n whisprx-prod deployment/whisprx

# Test health endpoint
kubectl port-forward -n whisprx-prod svc/whisprx 8000:80
curl http://localhost:8000/health/detailed
```

---

## Monitoring Setup

### Prometheus & Grafana

#### 1. Install Prometheus Operator

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
    -n monitoring --create-namespace
```

#### 2. Create ServiceMonitor

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: whisprx
  namespace: whisprx-prod
  labels:
    app: whisprx
spec:
  selector:
    matchLabels:
      app: whisprx
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

```bash
kubectl apply -f servicemonitor.yaml
```

#### 3. Create Grafana Dashboard

Import the WhispRX dashboard (ID: TBD) or use this PromQL:

**P95 Latency**:
```promql
histogram_quantile(0.95,
  rate(whisprx_latency_seconds_bucket[5m])
)
```

**Request Rate**:
```promql
rate(whisprx_requests_total[5m])
```

**Error Rate**:
```promql
rate(whisprx_errors_total[5m])
```

**GPU Utilization** (requires DCGM exporter):
```promql
DCGM_FI_DEV_GPU_UTIL{pod=~"whisprx.*"}
```

---

## Scaling

### Horizontal Pod Autoscaler

Enable HPA in `production-values.yaml`:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 75
```

### Custom Metrics (Queue Depth)

```yaml
# hpa-custom.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: whisprx-hpa
  namespace: whisprx-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: whisprx
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: whisprx_queue_size
      target:
        type: AverageValue
        averageValue: "5"  # Scale if queue > 5
```

---

## Security

### Network Policies

```yaml
# networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: whisprx-netpol
  namespace: whisprx-prod
spec:
  podSelector:
    matchLabels:
      app: whisprx
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443  # For model downloads
```

### Pod Security Standards

```yaml
# podsecurity.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: whisprx-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

---

## Disaster Recovery

### Backup Strategy

**Models**:
- Store in cloud object storage (S3, GCS)
- Version control with DVC or similar
- Download on pod startup

**Configuration**:
- All config in Git (GitOps)
- Helm values in version control

**Persistent Data**:
```bash
# Backup PVC
kubectl get pvc whisprx-models -n whisprx-prod -o yaml > pvc-backup.yaml

# Restore
kubectl apply -f pvc-backup.yaml
```

### Rollback

```bash
# Rollback to previous Helm release
helm rollback whisprx -n whisprx-prod

# Rollback to specific revision
helm rollback whisprx 2 -n whisprx-prod
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod -n whisprx-prod <pod-name>

# Common issues:
# 1. No GPU available
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# 2. Insufficient memory
kubectl top nodes

# 3. Image pull error
kubectl get events -n whisprx-prod --sort-by='.lastTimestamp'
```

### High Latency

```bash
# Check GPU utilization
kubectl exec -n whisprx-prod <pod-name> -- nvidia-smi

# Check queue sizes
curl http://<pod-ip>:8000/metrics | grep whisprx_queue_size

# Check resource throttling
kubectl top pod -n whisprx-prod
```

### Memory Leaks

```bash
# Monitor memory over time
kubectl top pod -n whisprx-prod -l app=whisprx --watch

# Restart pod
kubectl rollout restart deployment/whisprx -n whisprx-prod
```

---

## Cost Optimization

### Spot Instances

Use spot/preemptible nodes for non-production:

```yaml
nodeSelector:
  node.kubernetes.io/instance-type: n1-standard-4-spot

tolerations:
- key: cloud.google.com/gke-preemptible
  operator: Equal
  value: "true"
  effect: NoSchedule
```

### Resource Right-Sizing

Monitor actual usage:

```bash
# CPU/Memory usage
kubectl top pod -n whisprx-prod

# Adjust resources in values.yaml based on metrics
```

### Model Quantization

Use INT8 models to reduce memory:

```yaml
env:
  - name: LLM_QUANTIZATION
    value: "int8"
```

---

## Production Checklist

- [ ] NVIDIA Device Plugin installed
- [ ] Models pre-downloaded to PVC
- [ ] Helm values configured for production
- [ ] Resource limits set appropriately
- [ ] Health probes configured
- [ ] Ingress with TLS configured
- [ ] Prometheus metrics enabled
- [ ] Grafana dashboards created
- [ ] Autoscaling configured
- [ ] Network policies applied
- [ ] Backup strategy implemented
- [ ] Monitoring alerts configured
- [ ] Rollback procedure tested
- [ ] Load testing completed
- [ ] Documentation updated

---

## Performance Tuning

### GPU Optimization

```yaml
env:
  - name: CUDA_VISIBLE_DEVICES
    value: "0"
  - name: CUDA_LAUNCH_BLOCKING
    value: "0"  # Async kernel launches
```

### Model Optimization

```bash
# Precompile CUDA graphs on startup
kubectl exec -n whisprx-prod <pod-name> -- \
    python /app/scripts/precompile_cuda_graphs.py
```

### Network Optimization

```yaml
# Enable HTTP/2 for Ingress
ingress:
  annotations:
    nginx.ingress.kubernetes.io/http2-max-field-size: "16k"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
```

---

## Summary

WhispRX is now deployed to production with:
- ✅ GPU acceleration
- ✅ High availability
- ✅ Auto-scaling
- ✅ Monitoring & alerts
- ✅ Security hardening
- ✅ Disaster recovery

**Next steps**: Monitor performance, optimize costs, iterate on model selection.
