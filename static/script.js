document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const imagePreview = document.getElementById('image-preview');
  const previewArea = document.getElementById('preview-area');
  const loader = document.getElementById('loader');
  const resultsArea = document.getElementById('results-area');
  const resultClass = document.getElementById('result-class');
  const resultConfidence = document.getElementById('result-confidence');
  const confidenceLevel = document.getElementById('confidence-level');
  const toast = document.getElementById('toast');
  const resetButton = document.getElementById('reset-button');

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
      showToast('⚠️ La imagen es demasiado grande (máx. 5MB)');
      fileInput.value = '';
      return;
    }

    imagePreview.src = URL.createObjectURL(file);
    previewArea.style.display = 'block';
    loader.style.display = 'block';
    resultsArea.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
      .then(response => {
        loader.style.display = 'none';
        if (!response.ok) {
          return response.text().then(text => {
            throw new Error(text || 'Respuesta vacía del servidor');
          });
        }
        return response.json();
      })
      .then(data => {
        if (data.error) {
          resultClass.textContent = 'Error';
          resultConfidence.textContent = data.error;
          confidenceLevel.style.width = '0%';
          showToast('⚠️ Error: ' + data.error);
        } else {
          resultClass.textContent = data.class;
          resultConfidence.textContent = data.confidence;
          confidenceLevel.style.width = data.confidence;
          showToast('✅ Clasificación exitosa');
        }

        resultsArea.style.display = 'block';
      })
      .catch(error => {
        loader.style.display = 'none';
        resultClass.textContent = 'Error';
        resultConfidence.textContent = error.message || 'No se pudo procesar la imagen.';
        confidenceLevel.style.width = '0%';
        resultsArea.style.display = 'block';
        showToast('❌ ' + error.message);
        console.error('Error:', error);
      });
  });

  resetButton.addEventListener('click', () => {
    fileInput.value = '';
    imagePreview.src = '#';
    previewArea.style.display = 'none';
    resultsArea.style.display = 'none';
    confidenceLevel.style.width = '0%';
    resultClass.textContent = '';
    resultConfidence.textContent = '';
  });

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('visible');
    setTimeout(() =>