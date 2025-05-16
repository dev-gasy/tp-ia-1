# Étape de construction
FROM node:18-alpine as build

WORKDIR /app

# Copier les fichiers de dépendances
COPY package.json package-lock.json ./

# Installer les dépendances
RUN npm ci

# Copier le reste des fichiers de l'application
COPY . .

# Construire l'application
RUN npm run build

# Étape de production
FROM nginx:alpine

# Copier la configuration nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copier les fichiers de construction de l'application
COPY --from=build /app/dist /usr/share/nginx/html

# Exposer le port
EXPOSE 80

# Commande pour démarrer le serveur nginx
CMD ["nginx", "-g", "daemon off;"] 