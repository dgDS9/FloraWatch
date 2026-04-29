import { useState } from 'react';
import { motion, useMotionValue, useTransform, AnimatePresence } from 'motion/react';
import { SproutIcon, RefreshIcon, WaterDropIcon, CalendarIcon, SunIcon, SearchIcon } from '../Icons/Icons';
import PlantConfirmModal from '../PlantConfirmModal/PlantConfirmModal';
import usePlantImage from '../../hooks/usePlantImage';
import { useI18n } from '../../i18n/I18nContext';
import './ResultCard.css';

export default function ResultCard({ result, uncertain, onDismiss, onConfirmPlant }) {
  const { t, lang } = useI18n();
  const { url: plantImgUrl } = usePlantImage(uncertain ? null : result.name);
  const [confirmPlant, setConfirmPlant] = useState(null);

  /** Resolve a multilingual { en, de, es } value or plain string */
  const loc = (v) => (v && typeof v === 'object' ? (v[lang] || v.en) : v);

  const x = useMotionValue(0);
  const opacity = useTransform(x, [-150, 0, 150], [0.2, 1, 0.2]);
  const rotate = useTransform(x, [-150, 0, 150], [-8, 0, 8]);

  const handleDragEnd = (_, info) => {
    if (Math.abs(info.offset.x) > 100) onDismiss();
  };
  if (uncertain) {
    return (
      <motion.div
        className="result-card result-card--uncertain"
        style={{ x, opacity, rotate }}
        drag="x"
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.6}
        onDragEnd={handleDragEnd}
        initial={{ x: 60, opacity: 0, scale: 0.92 }}
        animate={{ x: 0, opacity: 1, scale: 1 }}
        exit={{ x: 200, opacity: 0, scale: 0.95 }}
        transition={{ type: 'spring', stiffness: 180, damping: 20 }}
      >
        <div className="result-header">
          <span className="result-badge result-badge--uncertain">{t('result.uncertain')}</span>
        </div>

        <h2 className="result-name">{t('result.couldNotIdentify')}</h2>
        <p className="result-description">{t('result.lowConfidence')}</p>

        <ul className="top3-list">
          {(result.top3 || []).map((guess, i) => (
            <li key={i} className="top3-item top3-item--clickable" role="button" tabIndex={0} onClick={() => setConfirmPlant(guess.name)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setConfirmPlant(guess.name); }}>
              <span className="top3-rank">#{i + 1}</span>
              <span className="top3-name">
                {guess.name}
                <SearchIcon size={12} color="#92400e" className="top3-search-icon" />
              </span>
              <div className="top3-bar-track">
                <motion.div
                  className="top3-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${guess.confidence}%` }}
                  transition={{ delay: 0.3 + i * 0.15, duration: 0.6, ease: 'easeOut' }}
                />
              </div>
              <span className="top3-pct">{guess.confidence}%</span>
            </li>
          ))}
        </ul>

        <button className="result-dismiss" onClick={onDismiss}>
          <RefreshIcon size={16} color="#fff" /> {t('result.tryAgain')}
        </button>

        <AnimatePresence>
          {confirmPlant && (
            <PlantConfirmModal
              plantName={confirmPlant}
              onConfirm={(name) => {
                setConfirmPlant(null);
                onConfirmPlant?.(name);
              }}
              onCancel={() => setConfirmPlant(null)}
            />
          )}
        </AnimatePresence>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="result-card"
      style={{ x, opacity, rotate }}
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.6}
      onDragEnd={handleDragEnd}
      initial={{ x: 60, opacity: 0, scale: 0.92 }}
      animate={{ x: 0, opacity: 1, scale: 1 }}
      exit={{ x: 200, opacity: 0, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 180, damping: 20 }}
    >
      <div className="result-header">
        <span className="result-badge"><SproutIcon size={14} color="#166534" /> {t('result.identified')}</span>
      </div>

      {plantImgUrl && (
        <motion.img
          src={plantImgUrl}
          alt={result.name}
          className="plant-illustration plant-illustration--clickable"
          onClick={() => setConfirmPlant(result.name)}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.15, duration: 0.4 }}
        />
      )}

      <h2 className="result-name">{result.name}</h2>
      {result.commonName && <p className="result-common-name">{loc(result.commonName)}</p>}

      <div className="result-confidence">
        <div className="confidence-bar-track">
          <motion.div
            className="confidence-bar-fill"
            initial={{ width: 0 }}
            animate={{ width: `${result.confidence}%` }}
            transition={{ delay: 0.3, duration: 0.8, ease: 'easeOut' }}
          />
        </div>
        <span className="confidence-label">{result.confidence}% {t('result.match')}</span>
      </div>

      {result.care && (
        <div className="care-tips">
          <div className="care-tip">
            <div className="care-icon-wrap care-icon-wrap--water">
              <WaterDropIcon size={22} />
            </div>
            <div className="care-text">
              <span className="care-label">{t('care.watering')}</span>
              <span className="care-value">{loc(result.care.water)}</span>
            </div>
          </div>
          <div className="care-tip">
            <div className="care-icon-wrap care-icon-wrap--freq">
              <CalendarIcon size={22} />
            </div>
            <div className="care-text">
              <span className="care-label">{t('care.frequency')}</span>
              <span className="care-value">{loc(result.care.frequency)}</span>
            </div>
          </div>
          <div className="care-tip">
            <div className="care-icon-wrap care-icon-wrap--sun">
              <SunIcon size={22} />
            </div>
            <div className="care-text">
              <span className="care-label">{t('care.sunlight')}</span>
              <span className="care-value">{loc(result.care.sunlight)}</span>
            </div>
          </div>
        </div>
      )}

      {result.toxicity && (
        <div className={`toxicity-badge toxicity-badge--${result.toxicity}`}>
          <span className="toxicity-icon">
            {result.toxicity === 'safe' ? '✓' : '⚠'}
          </span>
          <span className="toxicity-label">{t(`toxicity.${result.toxicity}`)}</span>
        </div>
      )}

      {result.funFact && (
        <div className="fun-fact">
          <span className="fun-fact-label">{t('fact.label')}</span>
          <p className="fun-fact-text">{loc(result.funFact)}</p>
        </div>
      )}

      {result.top3?.length > 0 && (
        <ul className="top3-list top3-list--success">
          {result.top3.map((guess, i) => (
            <li key={i} className="top3-item top3-item--clickable" role="button" tabIndex={0} onClick={() => setConfirmPlant(guess.name)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setConfirmPlant(guess.name); }}>
              <span className="top3-rank">#{i + 1}</span>
              <span className="top3-name">
                {guess.name}
                <SearchIcon size={12} color="#166534" className="top3-search-icon" />
              </span>
              <div className="top3-bar-track">
                <motion.div
                  className="top3-bar-fill top3-bar-fill--success"
                  initial={{ width: 0 }}
                  animate={{ width: `${guess.confidence}%` }}
                  transition={{ delay: 0.5 + i * 0.15, duration: 0.6, ease: 'easeOut' }}
                />
              </div>
              <span className="top3-pct top3-pct--success">{guess.confidence}%</span>
            </li>
          ))}
        </ul>
      )}

      <button className="result-dismiss" onClick={onDismiss}>
        <RefreshIcon size={16} color="#fff" /> {t('result.scanAgain')}
      </button>

      <AnimatePresence>
        {confirmPlant && (
          <PlantConfirmModal
            plantName={confirmPlant}
            onConfirm={(name) => {
              setConfirmPlant(null);
              onConfirmPlant?.(name);
            }}
            onCancel={() => setConfirmPlant(null)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
