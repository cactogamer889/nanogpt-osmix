const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // API calls
  getCategories: () => ipcRenderer.invoke('api-categories'),
  getCategoryStatus: (category, subcategory) => 
    ipcRenderer.invoke('api-category-status', category, subcategory),
  checkDataset: (category, subcategory) => 
    ipcRenderer.invoke('api-dataset-check', category, subcategory),
  startTraining: (config) => ipcRenderer.invoke('api-train', config),
  checkParams: (modelPath) => ipcRenderer.invoke('api-check-params', modelPath),
  retrainModel: (config) => ipcRenderer.invoke('api-retrain', config),
  getModels: () => ipcRenderer.invoke('api-models'),
  
  // Window controls
  minimizeWindow: () => ipcRenderer.invoke('window-minimize'),
  maximizeWindow: () => ipcRenderer.invoke('window-maximize'),
  closeWindow: () => ipcRenderer.invoke('window-close')
});
