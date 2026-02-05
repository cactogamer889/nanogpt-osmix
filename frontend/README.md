# Osmix Electron App

Aplicativo Electron para a plataforma de treinamento de IA Osmix.

## Instalação

```bash
npm install
```

## Executar

### Modo Desenvolvimento
```bash
npm run dev
```

### Modo Produção
```bash
npm start
```

## Build

Para criar executáveis:

```bash
npm run build
```

Os executáveis serão gerados na pasta `dist/`.

## Requisitos

- Node.js 16+
- Backend API rodando em `http://localhost:8000`

## Características

- Interface com fundo transparente e cinza
- Barra de título customizada
- Comunicação com API via IPC
- Suporte a todas as funcionalidades de treinamento
