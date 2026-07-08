"""
watch.py — Command-Line Interface (CLI) utility for CAMP.

Provides commands to start the FastAPI server, tail JSONL log files,
and inspect agent/alert metrics directly from the terminal.
"""

import os
import sys
import time
import json
import argparse
import requests
import uvicorn


def start_server(host: str, port: int):
    """Start the FastAPI API Server and Dashboard."""
    print(f"=== Starting CAMP Observability Server on http://{host}:{port} ===")
    uvicorn.run("camp.api.server:app", host=host, port=port, log_level="info")


def tail_log_file(filepath: str, endpoint: str):
    """Tail a JSONL log file and POST records to the observation API endpoint."""
    print(f"=== Tailing log file: '{filepath}' ===")
    print(f"Sending metrics payload to endpoint: {endpoint}")
    
    if not os.path.exists(filepath):
        print(f"Warning: File '{filepath}' does not exist yet. Waiting for creation...")
        # Create empty file to watch
        with open(filepath, "w") as f:
            pass

    with open(filepath, "r") as f:
        # Move pointer to the end of the file
        f.seek(0, os.SEEK_END)
        
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                    
                line = line.strip()
                if not line:
                    continue

                try:
                    # Check if valid JSON before sending
                    data = json.loads(line)
                    res = requests.post(endpoint, json=data, timeout=2.0)
                    if res.status_code == 200:
                        telemetry = res.json().get("telemetry", {})
                        surprise = telemetry.get("surprise", 0.0)
                        print(f" [Ingested] Agent: {data.get('agent_id')} | Cost: ${data.get('cost'):.4f} | Surprise: {surprise:.4f}")
                    else:
                        print(f" [Error] API returned status {res.status_code}: {res.text}")
                except json.JSONDecodeError:
                    print(f" [Skip] Invalid JSON format in line: '{line}'")
                except requests.exceptions.RequestException as e:
                    print(f" [Error] Failed to connect to server: {e}")
        except KeyboardInterrupt:
            print("\nTailing stopped.")


def list_alerts(server_url: str):
    """Query and print current active alerts from the API server."""
    endpoint = f"{server_url}/api/alerts"
    try:
        res = requests.get(endpoint, timeout=2.0)
        if res.status_code == 200:
            alerts = res.json()
            print(f"\n=== Current Active Alerts ({len(alerts)}) ===")
            print(f"{'Time':<12} | {'Agent ID':<15} | {'Metric':<10} | {'Alert Message':<40}")
            print("-" * 85)
            for a in alerts:
                time_str = time.strftime('%H:%M:%S', time.localtime(a['timestamp']))
                status = "RESOLVED" if a['resolved'] else "ACTIVE"
                print(f"{time_str:<12} | {a['agent_id']:<15} | {a['metric']:<10} | {a['message'][:40]} ({status})")
        else:
            print(f"Failed to query alerts: {res.status_code} - {res.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to server: {e}")


def main():
    parser = argparse.ArgumentParser(description="CAMP CLI Tooling")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: server
    server_parser = subparsers.add_parser("server", help="Start the dashboard server")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Command: watch-file
    watch_parser = subparsers.add_parser("watch-file", help="Tail a log file and stream metrics to server")
    watch_parser.add_argument("--file", required=True, help="Path to log JSONL file")
    watch_parser.add_argument("--server", default="http://127.0.0.1:8000", help="Server base URL")

    # Command: status
    status_parser = subparsers.add_parser("status", help="Get summary list of active alerts")
    status_parser.add_argument("--server", default="http://127.0.0.1:8000", help="Server base URL")

    args = parser.parse_args()

    if args.command == "server":
        start_server(args.host, args.port)
    elif args.command == "watch-file":
        endpoint = f"{args.server}/api/observe"
        tail_log_file(args.file, endpoint)
    elif args.command == "status":
        list_alerts(args.server)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
