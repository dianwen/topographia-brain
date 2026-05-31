# catan-brain: stateless Catanatron decision service for Topographia bot seats.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; catanatron is pure-Python (only networkx).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY catan_brain ./catan_brain

ENV PORT=8001
# One uvicorn process per container; the ProcessPoolExecutor inside holds the CPU
# parallelism (BRAIN_WORKERS ~= vCPUs). Do NOT also scale uvicorn workers. See docs/bot.md §6.
EXPOSE 8001
CMD ["sh", "-c", "uvicorn catan_brain.server:app --host 0.0.0.0 --port ${PORT}"]
