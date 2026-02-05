const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const axios = require('axios');

// API Base URL
const API_BASE = 'http://localhost:8000';

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: true
    },
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#323232',
      symbolColor: '#ffffff',
      height: 40
    }
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers for API calls
ipcMain.handle('api-categories', async () => {
  try {
    const response = await axios.get(`${API_BASE}/api/categories`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-category-status', async (event, category, subcategory) => {
  try {
    const response = await axios.get(`${API_BASE}/api/categories/${category}/${subcategory}/status`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-dataset-check', async (event, category, subcategory) => {
  try {
    const response = await axios.get(`${API_BASE}/api/datasets/${category}/${subcategory}`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-train', async (event, config) => {
  try {
    const response = await axios.post(`${API_BASE}/api/train`, config);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message 
    };
  }
});

ipcMain.handle('api-check-params', async (event, modelPath) => {
  try {
    const response = await axios.post(`${API_BASE}/api/check_params`, {
      model_path: modelPath
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message 
    };
  }
});

ipcMain.handle('api-retrain', async (event, config) => {
  try {
    const response = await axios.post(`${API_BASE}/api/retrain`, config);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message 
    };
  }
});

ipcMain.handle('api-models', async () => {
  try {
    const response = await axios.get(`${API_BASE}/api/models`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Window controls
ipcMain.handle('window-minimize', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.handle('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.handle('window-close', () => {
  if (mainWindow) mainWindow.close();
});
