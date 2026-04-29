import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SearchIcon, RefreshIcon } from '../Icons/Icons';
import PlantConfirmModal from '../PlantConfirmModal/PlantConfirmModal';
import { useI18n } from '../../i18n/I18nContext';
import './PlantOptionsCard.css';

export default function PlantOptionsCard({ result, onConfirm, onDismiss }) {
  const { t } = useI18n();
  const [confirmPlant, setConfirmPlant] = useState(null);

  return (
    <motion.div
      className="options-card"
      initial={{ x: 60, opacity: 0, scale: 0.92 }}
      animate={{ x: 0, opacity: 1, scale: 1 }}
      exit={{ x: 200, opacity: 0, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 180, damping: 20 }}
    >
      <h2 className="options-title">{t('options.title')}</h2>
      <p className="options-subtitle">{t('options.subtitle')}</p>

      <ul className="options-list">
        {(result.top3 || []).map((guess, i) => (
          <li
            key={i}
            className="options-item"
            role="button"
            tabIndex={0}
            onClick={() => setConfirmPlant(guess.name)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') setConfirmPlant(guess.name);
            }}
          >
            <span className="options-rank">#{i + 1}</span>
            <span className="options-name">
              {guess.name}
              <SearchIcon size={12} color="#3d6b3d" className="options-search-icon" />
            </span>
            <div className="options-bar-track">
              <motion.div
                className="options-bar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${guess.confidence}%` }}
                transition={{ delay: 0.3 + i * 0.15, duration: 0.6, ease: 'easeOut' }}
              />
            </div>
            <span className="options-pct">{guess.confidence}%</span>
          </li>
        ))}
      </ul>

      <button className="options-dismiss" onClick={onDismiss}>
        <RefreshIcon size={16} color="#fff" /> {t('result.tryAgain')}
      </button>

      <AnimatePresence>
        {confirmPlant && (
          <PlantConfirmModal
            plantName={confirmPlant}
            onConfirm={(name) => {
              setConfirmPlant(null);
              onConfirm?.(name);
            }}
            onCancel={() => setConfirmPlant(null)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
