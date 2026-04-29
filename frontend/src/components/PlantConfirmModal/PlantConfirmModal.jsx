import { createPortal } from 'react-dom';
import { motion } from 'motion/react';
import { useI18n } from '../../i18n/I18nContext';
import { getPlantCare } from '../../data/plantCare';
import usePlantImage from '../../hooks/usePlantImage';
import './PlantConfirmModal.css';

export default function PlantConfirmModal({ plantName, onConfirm, onCancel }) {
  const { t, lang } = useI18n();
  const care = getPlantCare(plantName);
  const commonName = care?.commonName?.[lang] || care?.commonName?.en || null;
  const { url: imgUrl, loading } = usePlantImage(plantName);

  return createPortal(
    <motion.div
      className="confirm-overlay"
      onClick={onCancel}
      initial={{ opacity: 0, pointerEvents: 'auto' }}
      animate={{ opacity: 1, pointerEvents: 'auto' }}
      exit={{ opacity: 0, pointerEvents: 'none' }}
    >
      <motion.div
        className="confirm-modal"
        onClick={(e) => e.stopPropagation()}
        initial={{ scale: 0.85, opacity: 0, y: 30 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.85, opacity: 0, y: 30 }}
        transition={{ type: 'spring', stiffness: 200, damping: 22 }}
      >
        <h3 className="confirm-plant-name">{plantName}</h3>
        {commonName && <p className="confirm-common-name">{commonName}</p>}

        <div className="confirm-img-wrap">
          {loading && <div className="confirm-spinner" />}
          {imgUrl && (
            <img
              src={imgUrl}
              alt={plantName}
              className="confirm-img"
            />
          )}
          {!loading && !imgUrl && (
            <p className="confirm-no-image">{t('confirm.noImage')}</p>
          )}
        </div>

        <p className="confirm-question">{t('confirm.question')}</p>

        <div className="confirm-actions">
          <button
            className="confirm-btn confirm-btn--yes"
            onClick={() => onConfirm(plantName)}
          >
            {t('confirm.yes')}
          </button>
          <button className="confirm-btn confirm-btn--no" onClick={onCancel}>
            {t('confirm.no')}
          </button>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}
