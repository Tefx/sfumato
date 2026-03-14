"""Allow running as: python -m frame_terminal [render|live|tv <IP>]"""
from .proto import main
import asyncio

asyncio.run(main())
