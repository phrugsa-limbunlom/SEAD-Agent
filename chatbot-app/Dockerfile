FROM node:20-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Install dependencies needed for building
RUN apk add --no-cache python3 make g++

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Clear npm cache and install dependencies
RUN npm cache clean --force && \
    rm -rf node_modules package-lock.json && \
    npm install --no-optional --legacy-peer-deps && \
    npm cache clean --force

# Copy all the source code to the container
COPY . .

# Build the React app for production
RUN npm run build

# Production stage
FROM nginx:alpine AS production

# Copy build output to nginx
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration if needed
# COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Development stage
FROM node:20-alpine AS development

WORKDIR /app

# Install dependencies needed for building
RUN apk add --no-cache python3 make g++

# Copy package.json and install dependencies
COPY package*.json ./

# Clear npm cache and install dependencies with specific flags
RUN npm cache clean --force && \
    rm -rf node_modules package-lock.json && \
    npm install --no-optional --legacy-peer-deps --no-audit --no-fund && \
    npm cache clean --force

# Copy source code
COPY . .

# Expose port 3000 for development server
EXPOSE 3000

# Start development server
CMD ["npm", "start"] 