"""Samsung The Frame TV connection and Art Mode control."""

from __future__ import annotations

import sys
from pathlib import Path


def test_connection(tv_ip: str, port: int = 8002) -> dict:
    """Test basic WebSocket connection to the TV."""
    results = {"ip": tv_ip, "port": port, "steps": {}}

    try:
        from samsungtvws import SamsungTVWS
    except ImportError:
        results["error"] = "samsungtvws not installed"
        return results

    # Step 1: Basic connection
    print(f"Connecting to {tv_ip}:{port}...")
    try:
        tv = SamsungTVWS(host=tv_ip, port=port, name="FrameTerminal")
        results["steps"]["connection"] = "ok"
        print("  ✓ Basic connection established")
    except Exception as e:
        results["steps"]["connection"] = f"failed: {e}"
        print(f"  ✗ Connection failed: {e}")
        return results

    # Step 2: Check if Art Mode is available
    print("Checking Art Mode availability...")
    try:
        art = tv.art()
        results["steps"]["art_mode_init"] = "ok"
        print("  ✓ Art Mode API initialized")
    except Exception as e:
        results["steps"]["art_mode_init"] = f"failed: {e}"
        print(f"  ✗ Art Mode API failed: {e}")
        return results

    # Step 3: Try to get art mode status
    print("Querying Art Mode status...")
    try:
        available = art.supported()
        results["steps"]["art_supported"] = f"ok: {available}"
        print(f"  ✓ Art Mode supported: {available}")
    except Exception as e:
        results["steps"]["art_supported"] = f"failed: {e}"
        print(f"  ⚠ Art Mode supported check failed: {e}")

    # Step 4: Try to list existing art
    print("Listing uploaded art...")
    try:
        art_list = art.available()
        count = len(art_list) if art_list else 0
        results["steps"]["art_list"] = f"ok: {count} items"
        print(f"  ✓ Found {count} art items")
    except Exception as e:
        results["steps"]["art_list"] = f"failed: {e}"
        print(f"  ⚠ Art listing failed: {e}")

    return results


def upload_art(tv_ip: str, image_path: Path, port: int = 8002) -> dict:
    """Upload an image to The Frame's Art Mode."""
    from samsungtvws import SamsungTVWS

    results = {"image": str(image_path)}

    print(f"Uploading {image_path.name} to {tv_ip}...")
    try:
        tv = SamsungTVWS(host=tv_ip, port=port, name="FrameTerminal")
        art = tv.art()

        with open(image_path, "rb") as f:
            data = f.read()

        # Upload
        art.upload(data, file_type="PNG", matte="none")
        results["upload"] = "ok"
        print(f"  ✓ Uploaded {image_path.name}")

    except Exception as e:
        results["upload"] = f"failed: {e}"
        print(f"  ✗ Upload failed: {e}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m frame_terminal.tv <TV_IP>")
        sys.exit(1)
    test_connection(sys.argv[1])
