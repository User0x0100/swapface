#!/bin/bash

yt-dlp -f "bestvideo[vcodec^=vp9]/bestvideo[vcodec^=avc1]"  "https://www.youtube.com/watch?v=r01qao0_Gng" --proxy "http://127.0.0.1:8080" --js-runtimes bun -o "%(id)s.%(ext)s"