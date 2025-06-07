#!/bin/sh
# lora_adapter_manager.sh - POSIX-compliant LoRA adapter management script
# Manages vLLM LoRA adapters with listing, loading, and history tracking

# Exit on any error
set -e

# Default configuration
DEFAULT_VLLM_API_URL="http://localhost:8000/v1/lora"
VLLM_API_URL="${VLLM_API_URL:-$DEFAULT_VLLM_API_URL}"
HISTORY_LOG="$HOME/adapter-history.log"

# Adapter directories to search
ADAPTER_DIRS="/opt/vllm/lora-adapters /var/lib/vllm/adapters $HOME/.vllm/adapters"

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

LoRA Adapter Manager for vLLM

OPTIONS:
    --list              List all .json and adapter files in configured directories
    --info              Query vLLM LoRA API status and display current adapters
    --load <adapter>    Load the specified adapter via vLLM API
    --history           Display adapter loading history
    --help              Show this help message

ENVIRONMENT:
    VLLM_API_URL       vLLM API endpoint (default: $DEFAULT_VLLM_API_URL)

ADAPTER DIRECTORIES:
EOF
    for dir in $ADAPTER_DIRS; do
        echo "    $dir"
    done
    echo ""
}

# List all adapter files in configured directories
list_adapters() {
    echo "Searching for adapters in configured directories..."
    found=0
    
    for dir in $ADAPTER_DIRS; do
        if [ -d "$dir" ]; then
            # Find .json files and other potential adapter files
            # Using find with -name for POSIX compliance
            adapters=$(find "$dir" -type f \( -name "*.json" -o -name "*.safetensors" -o -name "*.bin" \) 2>/dev/null || true)
            
            if [ -n "$adapters" ]; then
                echo "$adapters" | while IFS= read -r file; do
                    # Extract filename and parent directory
                    filename=$(basename "$file")
                    parent_dir=$(dirname "$file")
                    echo "$filename ($parent_dir)"
                    found=$((found + 1))
                done
            fi
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo "No adapter files found in any configured directory"
        return 1
    fi
}

# Query vLLM LoRA API for current status
query_api_info() {
    echo "Querying vLLM LoRA API at: $VLLM_API_URL"
    
    # Check if curl is available
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required but not installed" >&2
        return 1
    fi
    
    # Make API request with timeout
    response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" --connect-timeout 5 --max-time 10 "$VLLM_API_URL" 2>&1) || {
        echo "Error: Failed to connect to vLLM API at $VLLM_API_URL" >&2
        echo "Please ensure vLLM is running and accessible" >&2
        return 1
    }
    
    # Extract HTTP status code
    http_status=$(echo "$response" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    
    # Remove status line from response body
    body=$(echo "$response" | sed '/HTTP_STATUS:/d')
    
    echo "HTTP Status: $http_status"
    
    if [ "$http_status" -eq 200 ]; then
        echo "Response:"
        echo "$body" | {
            # Try to pretty-print JSON if jq is available
            if command -v jq >/dev/null 2>&1; then
                jq . 2>/dev/null || echo "$body"
            else
                echo "$body"
            fi
        }
    else
        echo "Error: API returned status $http_status" >&2
        [ -n "$body" ] && echo "Response: $body" >&2
        return 1
    fi
}

# Load an adapter via vLLM API
load_adapter() {
    adapter="$1"
    
    if [ -z "$adapter" ]; then
        echo "Error: No adapter specified" >&2
        return 1
    fi
    
    echo "Loading adapter: $adapter"
    
    # Check if curl is available
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required but not installed" >&2
        return 1
    fi
    
    # Prepare JSON payload
    json_payload="{\"adapter\":\"$adapter\"}"
    
    # Make POST request
    response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
        --connect-timeout 5 \
        --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$VLLM_API_URL" 2>&1) || {
        echo "Error: Failed to connect to vLLM API at $VLLM_API_URL" >&2
        return 1
    }
    
    # Extract HTTP status code
    http_status=$(echo "$response" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    
    # Remove status line from response body
    body=$(echo "$response" | sed '/HTTP_STATUS:/d')
    
    if [ "$http_status" -eq 200 ] || [ "$http_status" -eq 201 ] || [ "$http_status" -eq 204 ]; then
        echo "Success: Adapter loaded successfully (HTTP $http_status)"
        [ -n "$body" ] && echo "Response: $body"
        
        # Log successful load
        log_adapter_load "$adapter"
        return 0
    else
        echo "Error: Failed to load adapter (HTTP $http_status)" >&2
        [ -n "$body" ] && echo "Response: $body" >&2
        return 1
    fi
}

# Log adapter load to history file
log_adapter_load() {
    adapter="$1"
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Ensure history directory exists
    history_dir=$(dirname "$HISTORY_LOG")
    [ ! -d "$history_dir" ] && mkdir -p "$history_dir"
    
    # Append to history log
    echo "$timestamp ▶ Loaded adapter: $adapter" >> "$HISTORY_LOG" || {
        echo "Warning: Failed to write to history log" >&2
    }
}

# Display adapter loading history
show_history() {
    echo "Adapter Loading History ($HISTORY_LOG):"
    
    if [ -f "$HISTORY_LOG" ]; then
        cat "$HISTORY_LOG"
    else
        echo "No history found. History will be created after first adapter load."
    fi
}

# Main script logic
main() {
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    case "$1" in
        --list)
            list_adapters
            ;;
        --info)
            query_api_info
            ;;
        --load)
            if [ $# -lt 2 ]; then
                echo "Error: --load requires an adapter argument" >&2
                exit 1
            fi
            load_adapter "$2"
            ;;
        --history)
            show_history
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"