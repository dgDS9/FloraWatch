import { useRef, useEffect, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import { motion } from 'motion/react';
import { CameraIcon, FlashIcon } from '../Icons/Icons';
import './CameraScanner.css';

export default forwardRef(function CameraScanner({ scanning, frozen, previewUrl }, ref) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [error, setError] = useState(null);
  const [torchOn, setTorchOn] = useState(false);
  const [torchSupported, setTorchSupported] = useState(false);

  /* Expose captureFrame() to parent via ref */
  useImperativeHandle(ref, () => ({
    captureFrame: () => {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return null;
      const canvas = canvasRef.current || document.createElement('canvas');
      canvasRef.current = canvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      /* Pause the video to freeze the frame */
      video.pause();

      return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    },
  }), []);

  /* Resume video when unfrozen */
  useEffect(() => {
    const video = videoRef.current;
    if (!frozen && video && video.paused && video.srcObject) {
      video.play().catch(() => {});
    }
  }, [frozen]);

  const startCamera = useCallback(async () => {
    try {
      /* Use lower resolution on mobile to reduce GPU/memory pressure */
      const isMobile = /Mobi|Android/i.test(navigator.userAgent);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: isMobile ? 640 : 1280 },
          height: { ideal: isMobile ? 480 : 720 },
        },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraReady(true);
        /* Check torch support */
        const track = stream.getVideoTracks()[0];
        const caps = track.getCapabilities?.() || {};
        if (caps.torch) setTorchSupported(true);
      }
    } catch (err) {
      console.error('Camera access denied:', err);
      setError(
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access.'
          : 'Could not access camera. Check your device.'
      );
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, [startCamera]);

  const toggleTorch = useCallback(async () => {
    const track = videoRef.current?.srcObject?.getVideoTracks()[0];
    if (!track) return;
    const next = !torchOn;
    try {
      await track.applyConstraints({ advanced: [{ torch: next }] });
      setTorchOn(next);
    } catch {}
  }, [torchOn]);

  return (
    <motion.div
      className="scanner-panel"
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 120, damping: 18, delay: 0.2 }}
    >
      {/* Animated scan-line overlay */}
      {scanning && (
        <motion.div
          className="scan-line"
          initial={{ top: '0%' }}
          animate={{ top: ['0%', '100%', '0%'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      {/* Corner markers */}
      <div className="corner corner-tl" />
      <div className="corner corner-tr" />
      <div className="corner corner-bl" />
      <div className="corner corner-br" />

      {/* Torch toggle */}
      {torchSupported && (
        <button className={`torch-btn${torchOn ? ' torch-on' : ''}`} onClick={toggleTorch} aria-label="Toggle flashlight">
          <FlashIcon size={18} color={torchOn ? '#f5c842' : '#fff'} />
        </button>
      )}

      {error ? (
        <div className="camera-error">
          <span className="camera-error-icon"><CameraIcon size={40} color="#8aaa8a" /></span>
          <p>{error}</p>
          <button className="retry-btn" onClick={startCamera}>
            Try Again
          </button>
        </div>
      ) : (
        <>
          <video
            ref={videoRef}
            className="camera-feed"
            autoPlay
            playsInline
            muted
            style={{ display: previewUrl ? 'none' : 'block' }}
          />
          {previewUrl && (
            <img src={previewUrl} alt="Uploaded plant" className="camera-feed" />
          )}
        </>
      )}

      {!cameraReady && !error && (
        <div className="camera-loading">
          <motion.div
            className="camera-loading-dot"
            animate={{ scale: [1, 1.4, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.2, repeat: Infinity }}
          />
          <span>Starting camera…</span>
        </div>
      )}
    </motion.div>
  );
});
