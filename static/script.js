document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const uploadText = document.getElementById('upload-text');
  const imagePreview = document.getElementById('image-preview');
  const previewArea = document.getElementById('preview-area');
  const loader = document.getElementById('loader');
  const resultsArea = document.getElementById('results-area');
  const resultClass = document.getElementById('result-class');
  const resultConfidence = document.getElementById('result-confidence');
  const confidenceLevel = document.getElementById('confidence-level');
  const toast = document.getElementById('toast');

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;

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
        if (!response.ok) {
          throw new Error('Respuesta no válida del servidor');
        }
        return response.json();
      })
      .then(data => {
        loader.style.display = 'none';

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
        resultConfidence.textContent = 'No se pudo procesar la imagen.';
        confidenceLevel.style.width = '0%';
        resultsArea.style.display = 'block';
        showToast('❌ Error al procesar la imagen');
        console.error('Error:', error);
      });
  });

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('visible');
    setTimeout(() => {
      toast.classList.remove('visible');
    }, 4000);
  }
});
