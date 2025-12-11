async function uploadVideo() {
    const file = document.getElementById('video').files[0];
    if (!file) return alert('Upload a video.');
    const form = new FormData();
    form.append('video', file);
    const res = await fetch('http://localhost:5000/analyze', { method: 'POST', body: form });
    const data = await res.json();
    document.getElementById('response').textContent = data.advice || 'Error';
}