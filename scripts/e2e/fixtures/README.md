# Place canned MP4 clips here for the `simulate-clip` command.
# Suggested fixtures (generate with ffmpeg):
#
#   ffmpeg -f lavfi -i testsrc=s=640x480:r=15:d=15 -pix_fmt yuv420p -t 15 fall.mp4
#   ffmpeg -f lavfi -i color=c=black:s=640x480:r=15:d=15 -pix_fmt yuv420p -t 15 quiet.mp4
#
# These are NOT shipped in git (likely large + binary) — add the actual
# MP4 files manually after running the ffmpeg commands above.
