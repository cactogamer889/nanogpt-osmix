// Current selected category and subcategory
let currentCategory = null;
let currentSubcategory = null;

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    await loadCategories();
    setupEventListeners();
    showSection('train-section'); // Show train section by default
});

// Show section and update active button
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Update active button
    document.querySelectorAll('.nav-main-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const buttonMap = {
        'train-section': 'btn-train',
        'check-params-section': 'btn-check-params',
        'retrain-section': 'btn-retrain'
    };
    
    if (buttonMap[sectionId]) {
        document.getElementById(buttonMap[sectionId]).classList.add('active');
    }
}

// Load categories from API using Electron IPC
async function loadCategories() {
    try {
        const result = await window.electronAPI.getCategories();
        if (!result.success) {
            console.error('Error loading categories:', result.error);
            return;
        }
        const data = result.data;
        
        const trainCategoriesDiv = document.getElementById('train-categories');
        trainCategoriesDiv.innerHTML = '';
        
        for (const [category, subcategories] of Object.entries(data.categories)) {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'category-item';
            categoryDiv.textContent = category;
            categoryDiv.onclick = () => toggleCategory(category, categoryDiv);
            
            const subcategoriesDiv = document.createElement('div');
            subcategoriesDiv.className = 'subcategories';
            subcategoriesDiv.style.display = 'none';
            
            for (const subcategory of subcategories) {
                const subcategoryDiv = document.createElement('div');
                subcategoryDiv.className = 'subcategory-item';
                subcategoryDiv.textContent = subcategory;
                
                // Check if available
                checkSubcategoryStatus(category, subcategory).then(result => {
                    if (result.success && result.data.available) {
                        subcategoryDiv.classList.add('available');
                        subcategoryDiv.onclick = (e) => {
                    // Remove active from all subcategories in same category
                    const parent = subcategoryDiv.closest('.subcategories');
                    if (parent) {
                        parent.querySelectorAll('.subcategory-item').forEach(item => {
                            item.classList.remove('active');
                        });
                    }
                    // Add active to clicked subcategory
                    subcategoryDiv.classList.add('active');
                    openTrainingConfig(category, subcategory);
                };
                    } else {
                        subcategoryDiv.classList.add('coming-soon');
                        subcategoryDiv.textContent += ' (Em breve)';
                    }
                });
                
                subcategoriesDiv.appendChild(subcategoryDiv);
            }
            
            categoryDiv.appendChild(subcategoriesDiv);
            trainCategoriesDiv.appendChild(categoryDiv);
        }
        
        // Load categories for retrain
        loadRetrainCategories(data.categories);
    } catch (error) {
        console.error('Error loading categories:', error);
    }
}

// Check subcategory status using Electron IPC
async function checkSubcategoryStatus(category, subcategory) {
    try {
        return await window.electronAPI.getCategoryStatus(category, subcategory);
    } catch (error) {
        return { success: false, data: { available: false } };
    }
}

// Toggle category
function toggleCategory(category, element) {
    const subcategories = element.querySelector('.subcategories');
    const isActive = element.classList.contains('active');
    
    // Close all categories
    document.querySelectorAll('.category-item').forEach(item => {
        item.classList.remove('active');
        item.querySelector('.subcategories').style.display = 'none';
    });
    
    // Toggle current
    if (!isActive) {
        element.classList.add('active');
        subcategories.style.display = 'block';
    }
}

// Open training configuration modal
async function openTrainingConfig(category, subcategory) {
    currentCategory = category;
    currentSubcategory = subcategory;
    
    // Check dataset availability using Electron IPC
    try {
        const result = await window.electronAPI.checkDataset(category, subcategory);
        if (result.success && !result.data.ready) {
            alert(`Dataset files not found. Please ensure train.bin and val.bin exist in:\n${result.data.dataset_path}`);
            return;
        }
    } catch (error) {
        console.error('Error checking dataset:', error);
    }
    
    // Show modal
    document.getElementById('config-modal').classList.add('show');
}

// Close config modal
function closeConfigModal() {
    document.getElementById('config-modal').classList.remove('show');
}

// Setup event listeners
function setupEventListeners() {
    // Training config form
    document.getElementById('training-config-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await startTraining();
    });
    
    // Close modal on outside click
    document.getElementById('config-modal').addEventListener('click', (e) => {
        if (e.target.id === 'config-modal') {
            closeConfigModal();
        }
    });
}

// Start training
async function startTraining() {
    const config = {
        category: currentCategory,
        subcategory: currentSubcategory,
        model_name: document.getElementById('config-model-name').value,
        batch_size: parseInt(document.getElementById('config-batch-size').value),
        block_size: parseInt(document.getElementById('config-block-size').value),
        gradient_accumulation_steps: parseInt(document.getElementById('config-gradient-accumulation').value),
        learning_rate: parseFloat(document.getElementById('config-learning-rate').value),
        max_iters: parseInt(document.getElementById('config-max-iters').value),
        n_layer: parseInt(document.getElementById('config-n-layer').value),
        n_head: parseInt(document.getElementById('config-n-head').value),
        n_embd: parseInt(document.getElementById('config-n-embd').value),
        dropout: parseFloat(document.getElementById('config-dropout').value),
        output_format: document.getElementById('config-output-format').value,
        device: document.getElementById('config-device').value,
        dtype: document.getElementById('config-dtype').value,
        save_interval: parseInt(document.getElementById('config-save-interval').value)
    };
    
    try {
        const result = await window.electronAPI.startTraining(config);
        
        if (result.success) {
            alert(`Training started successfully!\nTraining ID: ${result.data.training_id}`);
            closeConfigModal();
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        alert(`Error starting training: ${error.message}`);
    }
}

// Check parameters
async function checkParams() {
    const modelPath = document.getElementById('model-path').value;
    
    if (!modelPath) {
        alert('Please enter a model path');
        return;
    }
    
    try {
        const result = await window.electronAPI.checkParams(modelPath);
        const resultDiv = document.getElementById('params-result');
        
        if (result.success) {
            resultDiv.innerHTML = `
                <h3 style="color: #333; margin-bottom: 10px;">Model Parameters</h3>
                <pre>${JSON.stringify(result.data, null, 2)}</pre>
            `;
            resultDiv.classList.add('show');
        } else {
            resultDiv.innerHTML = `<div class="message error">Error: ${result.error}</div>`;
            resultDiv.classList.add('show');
        }
    } catch (error) {
        const resultDiv = document.getElementById('params-result');
        resultDiv.innerHTML = `<div class="message error">Error: ${error.message}</div>`;
        resultDiv.classList.add('show');
    }
}

// Load categories for retrain
function loadRetrainCategories(categories) {
    const categorySelect = document.getElementById('retrain-category');
    categorySelect.innerHTML = '<option value="">Select Category</option>';
    
    for (const category of Object.keys(categories)) {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        categorySelect.appendChild(option);
    }
    
    categorySelect.addEventListener('change', () => {
        const subcategorySelect = document.getElementById('retrain-subcategory');
        subcategorySelect.innerHTML = '<option value="">Select Subcategory</option>';
        
        if (categorySelect.value) {
            for (const subcategory of categories[categorySelect.value]) {
                const option = document.createElement('option');
                option.value = subcategory;
                option.textContent = subcategory;
                subcategorySelect.appendChild(option);
            }
        }
    });
}

// Retrain model
async function retrainModel() {
    const config = {
        model_path: document.getElementById('retrain-model-path').value,
        category: document.getElementById('retrain-category').value,
        subcategory: document.getElementById('retrain-subcategory').value,
        additional_iters: parseInt(document.getElementById('additional-iters').value)
    };
    
    if (!config.model_path || !config.category || !config.subcategory) {
        alert('Please fill all fields');
        return;
    }
    
    try {
        const result = await window.electronAPI.retrainModel(config);
        const resultDiv = document.getElementById('retrain-result');
        
        if (result.success) {
            resultDiv.innerHTML = `
                <div class="message success">
                    Retraining started successfully!<br>
                    Retrain ID: ${result.data.retrain_id}<br>
                    Additional Iterations: ${result.data.additional_iters}
                </div>
            `;
            resultDiv.classList.add('show');
        } else {
            resultDiv.innerHTML = `<div class="message error">Error: ${result.error}</div>`;
            resultDiv.classList.add('show');
        }
    } catch (error) {
        const resultDiv = document.getElementById('retrain-result');
        resultDiv.innerHTML = `<div class="message error">Error: ${error.message}</div>`;
        resultDiv.classList.add('show');
    }
}
