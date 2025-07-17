# Use an official Node.js image
FROM node:16-alpine AS development

# Set working directory
WORKDIR /app

# Copy package files first to leverage caching
COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]