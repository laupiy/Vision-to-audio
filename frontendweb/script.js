document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const btnToggleMode = document.getElementById('btn-toggle-mode');
    const btnReplayAudio = document.getElementById('btn-replay-audio');
    const btnSourceServer = document.getElementById('btn-source-server');
    const btnSourceDevice = document.getElementById('btn-source-device');
    
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const modeText = document.getElementById('mode-text');
    
    const videoFeed = document.getElementById('video-feed');
    const deviceVideo = document.getElementById('device-video');
    const processedFrame = document.getElementById('processed-frame');
    const captureCanvas = document.getElementById('capture-canvas');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    
    const resultLabel = document.getElementById('result-label');
    const audioWave = document.getElementById('audio-wave');

    // Sliders
    const sliders = {
        brightness: document.getElementById('slider-brightness'),
        contrast: document.getElementById('slider-contrast'),
        confidence: document.getElementById('slider-confidence'),
        overlap: document.getElementById('slider-overlap'),
        opacity: document.getElementById('slider-opacity')
    };

    const sliderValues = {
        brightness: document.getElementById('val-brightness'),
        contrast: document.getElementById('val-contrast'),
        confidence: document.getElementById('val-confidence'),
        overlap: document.getElementById('val-overlap'),
        opacity: document.getElementById('val-opacity')
    };

    // State
    let isCameraOn = false;
    let isGuideMode = true;
    let pollingInterval = null;
    
    // Camera source: 'server' or 'device'
    let cameraSource = 'server';
    let deviceStream = null;
    let deviceLoopId = null;
    let isProcessing = false; // prevent concurrent requests

    // Initialize Slider Values
    function updateSliderValue(key) {
        let val = sliders[key].value;
        if (key === 'brightness' || key === 'contrast') {
            sliderValues[key].textContent = val + '%';
        } else {
            sliderValues[key].textContent = (val / 100).toFixed(2);
        }
    }

    Object.keys(sliders).forEach(key => {
        updateSliderValue(key);
        sliders[key].addEventListener('input', () => {
            updateSliderValue(key);
            sendParameterUpdate(key, sliders[key].value);
        });
    });

    // Backend Connection
    async function sendParameterUpdate(parameter, value) {
        try {
            await fetch('/update_params', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [parameter]: value })
            });
        } catch (err) {
            console.error('Failed to update parameter:', err);
        }
    }

    // ============================================================
    // SOURCE TOGGLE: Server Camera vs Device Camera
    // ============================================================
    btnSourceServer.addEventListener('click', () => {
        if (cameraSource === 'server') return;
        cameraSource = 'server';
        btnSourceServer.classList.add('active-source');
        btnSourceDevice.classList.remove('active-source');
        
        // If camera is currently on, switch source live
        if (isCameraOn) {
            stopDeviceCamera();
            startServerCamera();
        }
    });

    btnSourceDevice.addEventListener('click', () => {
        if (cameraSource === 'device') return;
        cameraSource = 'device';
        btnSourceDevice.classList.add('active-source');
        btnSourceServer.classList.remove('active-source');
        
        // If camera is currently on, switch source live
        if (isCameraOn) {
            stopServerCamera();
            startDeviceCamera();
        }
    });

    // ============================================================
    // CAMERA TOGGLE
    // ============================================================
    btnToggleCamera.addEventListener('click', async () => {
        isCameraOn = !isCameraOn;

        if (isCameraOn) {
            statusDot.classList.replace('offline', 'online');
            statusText.textContent = 'Kamera Aktif';
            
            btnToggleCamera.classList.add('danger');
            btnToggleCamera.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="1" y1="1" x2="23" y2="23"></line><path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34m-7.72-2.06a4 4 0 1 1-5.56-5.56"></path></svg>
                <span>Matikan Kamera</span>
            `;

            cameraPlaceholder.style.display = 'none';

            if (cameraSource === 'server') {
                await startServerCamera();
            } else {
                await startDeviceCamera();
            }
        } else {
            statusDot.classList.replace('online', 'offline');
            statusText.textContent = 'Kamera Mati';
            
            btnToggleCamera.classList.remove('danger');
            btnToggleCamera.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 7l-7 5 7 5V7z"></path><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
                <span>Nyalakan Kamera</span>
            `;

            stopServerCamera();
            stopDeviceCamera();

            cameraPlaceholder.style.display = 'flex';
            resultLabel.textContent = 'Belum ada deteksi';
            resultLabel.style.color = 'white';
            stopAudioWave();
            stopPollingStatus();
        }
    });

    // ============================================================
    // SERVER CAMERA (MJPEG stream from Flask)
    // ============================================================
    async function startServerCamera() {
        // Tell server to turn on its camera
        try {
            await fetch('/toggle_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: true })
            });
        } catch (err) {
            console.error('Failed to start server camera:', err);
        }

        deviceVideo.style.display = 'none';
        processedFrame.style.display = 'none';
        videoFeed.style.display = 'block';
        videoFeed.src = '/video_feed?' + new Date().getTime();

        startPollingStatus();
    }

    function stopServerCamera() {
        videoFeed.style.display = 'none';
        videoFeed.src = '';

        // Tell server to turn off its camera
        fetch('/toggle_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: false })
        }).catch(() => {});
    }

    // ============================================================
    // DEVICE CAMERA (getUserMedia -> send frames to server)
    // ============================================================
    async function startDeviceCamera() {
        try {
            // Request camera with environment-facing (rear) camera preferred on mobile
            const constraints = {
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };

            deviceStream = await navigator.mediaDevices.getUserMedia(constraints);
            deviceVideo.srcObject = deviceStream;
            
            // Show the live preview from device (with slight transparency for layering)
            videoFeed.style.display = 'none';
            deviceVideo.style.display = 'block';
            processedFrame.style.display = 'block';
            
            // Wait for video to be ready
            deviceVideo.onloadedmetadata = () => {
                captureCanvas.width = deviceVideo.videoWidth;
                captureCanvas.height = deviceVideo.videoHeight;
                // Start capture-and-send loop
                startDeviceCaptureLoop();
            };

        } catch (err) {
            console.error('Gagal mengakses kamera device:', err);
            statusText.textContent = 'Gagal akses kamera';
            resultLabel.textContent = 'Kamera device tidak tersedia. Pastikan menggunakan HTTPS.';
            resultLabel.style.color = 'var(--status-offline)';
        }
    }

    function stopDeviceCamera() {
        if (deviceLoopId) {
            clearInterval(deviceLoopId);
            deviceLoopId = null;
        }

        if (deviceStream) {
            deviceStream.getTracks().forEach(track => track.stop());
            deviceStream = null;
        }

        deviceVideo.style.display = 'none';
        deviceVideo.srcObject = null;
        processedFrame.style.display = 'none';
        processedFrame.src = '';
    }

    function startDeviceCaptureLoop() {
        // Send a frame to server every ~300ms (adjustable)
        const INTERVAL_MS = 300;

        deviceLoopId = setInterval(async () => {
            if (!isCameraOn || cameraSource !== 'device' || isProcessing) return;
            
            isProcessing = true;

            try {
                const ctx = captureCanvas.getContext('2d');
                ctx.drawImage(deviceVideo, 0, 0, captureCanvas.width, captureCanvas.height);
                
                // Get JPEG base64 with reduced quality for speed
                const frameData = captureCanvas.toDataURL('image/jpeg', 0.6);

                const res = await fetch('/process_client_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame: frameData })
                });

                const data = await res.json();
                const currentLabel = data.label;
                
                // Show the annotated frame from server
                if (data.frame) {
                    processedFrame.src = data.frame;
                }

                // Update label and dashboard
                resultLabel.textContent = currentLabel;
                const hsvVal = document.getElementById('hsv-value');
                const tplVal = document.getElementById('template-score-value');
                if(hsvVal) hsvVal.textContent = data.hsv || "-";
                if(tplVal) tplVal.textContent = data.template_score || "-";

                if (currentLabel.startsWith("Rp")) {
                    resultLabel.style.color = "var(--status-online)";
                    playAudioWave();
                } else if (currentLabel === "Cahaya Kurang") {
                    resultLabel.style.color = "var(--status-offline)";
                } else {
                    resultLabel.style.color = "white";
                }

            } catch (err) {
                console.error('Error sending frame:', err);
            } finally {
                isProcessing = false;
            }

        }, INTERVAL_MS);
    }

    // ============================================================
    // RESET SETTINGS
    // ============================================================
    const btnResetSettings = document.getElementById('btn-reset-settings');
    if (btnResetSettings) {
        btnResetSettings.addEventListener('click', () => {
            sliders.brightness.value = 50;
            sliders.contrast.value = 50;
            updateSliderValue('brightness');
            updateSliderValue('contrast');
            sendParameterUpdate('brightness', 50);
            sendParameterUpdate('contrast', 50);
        });
    }

    // ============================================================
    // AUDIO WAVE
    // ============================================================
    function playAudioWave() {
        audioWave.classList.add('playing');
        setTimeout(stopAudioWave, 2000);
    }

    function stopAudioWave() {
        audioWave.classList.remove('playing');
    }

    btnReplayAudio.addEventListener('click', async () => {
        if (resultLabel.textContent !== 'Belum ada deteksi') {
            playAudioWave();
            fetch('/replay_audio', { method: 'POST' });
        }
    });

    // ============================================================
    // POLLING STATUS (for server camera mode only)
    // ============================================================
    function startPollingStatus() {
        if (pollingInterval) clearInterval(pollingInterval);
        
        let lastLabel = "";
        
        pollingInterval = setInterval(async () => {
            if (!isCameraOn || cameraSource !== 'server') return;
            
            try {
                const res = await fetch('/status');
                const data = await res.json();
                
                const currentLabel = data.label;
                resultLabel.textContent = currentLabel;
                
                const hsvVal = document.getElementById('hsv-value');
                const tplVal = document.getElementById('template-score-value');
                if(hsvVal) hsvVal.textContent = data.hsv || "-";
                if(tplVal) tplVal.textContent = data.template_score || "-";

                if (currentLabel.startsWith("Rp")) {
                    resultLabel.style.color = "var(--status-online)";
                    if (currentLabel !== lastLabel) {
                        playAudioWave();
                    }
                } else if (currentLabel === "Cahaya Kurang" || currentLabel === "Kamera Mati") {
                    resultLabel.style.color = "var(--status-offline)";
                } else {
                    resultLabel.style.color = "white";
                }
                
                lastLabel = currentLabel;
            } catch (err) {
                console.error("Error fetching status:", err);
            }
        }, 500);
    }
    
    function stopPollingStatus() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }
});
