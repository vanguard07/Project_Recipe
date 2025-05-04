FROM node:22-alpine

WORKDIR /app

# Copy package.json and package-lock.json first for better cache utilization
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Expose the port Next.js runs on
EXPOSE 3000

# Run in development mode instead of requiring a build
CMD ["npm", "run", "dev"]