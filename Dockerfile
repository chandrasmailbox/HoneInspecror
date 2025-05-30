# Stage 1: Build React App
FROM node:20 AS frontend-build
ARG FRONTEND_ENV
ENV FRONTEND_ENV=${FRONTEND_ENV}
WORKDIR /app
COPY frontend/ /app/
RUN rm -f /app/.env
RUN touch /app/.env
RUN echo "${FRONTEND_ENV}" | tr ',' '\n' > /app/.env
RUN cat /app/.env
RUN yarn install --frozen-lockfile && yarn build

# Stage 2: Build Python Backend
FROM python:3.10-slim AS backend
WORKDIR /app
COPY backend/ /app/
RUN rm -f /app/.env
COPY backend/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final Image
FROM nginx:stable-alpine
# Copy built frontend
COPY --from=frontend-build /app/build /usr/share/nginx/html
# Copy backend
COPY --from=backend /app /backend
# Copy nginx config and entrypoint
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# If you need to run the backend, do so with python3 from /backend
# Alpine's python is not used for pip installs; all python code is in /backend from the python:3.10-slim build

ENV PYTHONUNBUFFERED=1

# Start both services: Uvicorn and Nginx (adjust entrypoint.sh as needed)
CMD ["/entrypoint.sh"]