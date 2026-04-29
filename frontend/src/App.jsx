import { useState, useCallback, useRef, useEffect, lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import CameraScanner from './components/CameraScanner/CameraScanner';
import { VineBehind, VineFront } from './components/VineDecoration/VineDecoration';
import ScanButton from './components/ScanButton/ScanButton';
import StatusMessage from './components/StatusMessage/StatusMessage';
import SkeletonCard from './components/SkeletonCard/SkeletonCard';
import { FloraLogoIcon, GalleryIcon, InfoIcon } from './components/Icons/Icons';
import { warmupBackend, predictPlant } from './services/api';
import { getPlantCare } from './data/plantCare';
import { getRandomFact } from './data/plantFacts';
import { getHistory, addToHistory, clearHistory } from './services/history';
import PlantOptionsCard from './components/PlantOptionsCard/PlantOptionsCard';
import LangPicker from './components/LangPicker/LangPicker';
import PlantListModal from './components/PlantListModal/PlantListModal';
import './components/PlantListModal/PlantListModal.css';
import './App.css';

/* Lazy-load heavy components that aren't needed on initial render */
const ResultCard = lazy(() => import('./components/ResultCard/ResultCard'));
const ScanHistory = lazy(() => import('./components/ScanHistory/ScanHistory'));

/* Short haptic pulse — safe no-op on unsupported devices */
const vibrate = (pattern = 15) => {
  try { navigator.vibrate?.(pattern); } catch {}
};

export default function App() {
  const [status, setStatus] = useState('idle'); // idle | scanning | success | error
  const [result, setResult] = useState(null);
  const [showFlowers, setShowFlowers] = useState(false);
  const [showPlantList, setShowPlantList] = useState(false);
  const [history, setHistory] = useState(() => getHistory());
  const [previewUrl, setPreviewUrl] = useState(null);
  const scannerRef = useRef(null);
  const fileInputRef = useRef(null);

  /* Wake up the backend as soon as the app loads */
  useEffect(() => { warmupBackend(); }, []);

  /* Shared prediction handler for both camera & upload */
  const processBlob = useCallback(async (blob) => {
    vibrate(15);
    setStatus('scanning');
    setResult(null);
    setShowFlowers(false);

    try {
      let plant = await predictPlant(blob);

      setResult(plant);
      vibrate([10, 40, 10]);
      setStatus('picking');
    } catch {
      vibrate([30, 50, 30]);
      setStatus('error');
    }
  }, []);

  const handleScan = useCallback(async () => {
    const blob = scannerRef.current ? await scannerRef.current.captureFrame() : null;
    processBlob(blob);
  }, [processBlob]);

  const handleUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    processBlob(file);
    e.target.value = '';
  }, [processBlob]);

  const handleDismiss = useCallback(() => {
    setStatus('idle');
    setResult(null);
    setShowFlowers(false);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  const handleClearHistory = useCallback(() => {
    clearHistory();
    setHistory([]);
  }, []);

  const handleConfirmPlant = useCallback((plantName) => {
    const care = getPlantCare(plantName);
    const confirmedResult = {
      unknown: false,
      name: plantName,
      commonName: care?.commonName || null,
      confidence: result?.top3?.find((g) => g.name === plantName)?.confidence || 0,
      care: care
        ? { water: care.water, frequency: care.frequency, sunlight: care.sunlight }
        : null,
      toxicity: care?.toxicity || null,
      funFact: getRandomFact(plantName),
      top3: result?.top3 || [],
    };
    setResult(confirmedResult);
    vibrate([10, 30, 10, 30, 10]);
    setStatus('success');
    setShowFlowers(true);
    setHistory(addToHistory(confirmedResult));
  }, [result]);

  return (
    <div className="app">
      <LangPicker />

      {/* Info button – top left */}
      <button
        className="info-btn-corner"
        onClick={() => setShowPlantList(true)}
        aria-label="Show supported plants"
      >
        <InfoIcon size={15} color="#3d8b3d" />
      </button>

      {/* Title */}
      <motion.h1
        className="app-title"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 140, damping: 16 }}
      >
        <span className="title-icon"><FloraLogoIcon size={32} /></span>
        <span>Flora<span className="title-accent">Watch</span></span>
      </motion.h1>

      {/* Status */}
      <StatusMessage status={status} isFirstScan={history.length === 0} />

      {/* Scanner area – DOM order controls layering:
           behind vines → scanner → front vines */}
      <div className="scanner-area">
        <VineBehind side="left" />
        <VineBehind side="right" />
        <CameraScanner ref={scannerRef} scanning={status === 'scanning'} frozen={status !== 'idle'} previewUrl={previewUrl} />
        <VineFront side="left" showFlowers={showFlowers} />
        <VineFront side="right" showFlowers={showFlowers} />
      </div>

      {/* Action / Result */}
      <div className="action-area">
        <AnimatePresence mode="wait">
          {(() => {
            if (status === 'scanning') {
              return <SkeletonCard key="skeleton" />;
            }
            if (status === 'picking' && result) {
              return <PlantOptionsCard key="options" result={result} onConfirm={handleConfirmPlant} onDismiss={handleDismiss} />;
            }
            if (result) {
              return (
                <Suspense fallback={<SkeletonCard />}>
                  <ResultCard key="result" result={result} onDismiss={handleDismiss} onConfirmPlant={handleConfirmPlant} />
                </Suspense>
              );
            }
            return (
              <motion.div
                key="scan"
                className="scan-actions"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.25 }}
              >
                <ScanButton
                  onClick={handleScan}
                  disabled={status === 'scanning'}
                  scanning={status === 'scanning'}
                />
                <button
                  className="upload-btn"
                  onClick={() => fileInputRef.current?.click()}
                  aria-label="Upload photo"
                >
                  <GalleryIcon size={20} color="#3d8b3d" />
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="upload-input"
                  onChange={handleUpload}
                />
              </motion.div>
            );
          })()}
        </AnimatePresence>
      </div>

      {/* Scan history */}
      {status === 'idle' && (
        <Suspense fallback={null}>
          <ScanHistory history={history} onClear={handleClearHistory} />
        </Suspense>
      )}

      {/* Plant list modal */}
      <PlantListModal open={showPlantList} onClose={() => setShowPlantList(false)} />
    </div>
  );
}
