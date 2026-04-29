import { useState, useCallback } from 'react';
import { motion, AnimatePresence, useMotionValue, animate } from 'motion/react';
import { useI18n } from '../../i18n/I18nContext';
import { getPlantList, getPlantCare } from '../../data/plantCare';
import { getRandomFact, getNextFact } from '../../data/plantFacts';
import usePlantImage from '../../hooks/usePlantImage';
import { LeafIcon, WaterDropIcon, CalendarIcon, SunIcon, RefreshIcon } from '../Icons/Icons';
import './PlantListModal.css';

/* ─── Horizontal drag-to-dismiss (swipe left to close) ─── */
function useDragXToClose(onClose) {
  const x = useMotionValue(0);

  const handleDragEnd = useCallback((_, info) => {
    if (info.offset.x < -80 || info.velocity.x < -300) {
      animate(x, -400, { duration: 0.2 });
      onClose();
    } else {
      animate(x, 0, { type: 'spring', stiffness: 300, damping: 28 });
    }
  }, [onClose, x]);

  return { x, handleDragEnd };
}

/* ─── Thumbnail ─── */
function PlantThumb({ name }) {
  const { url, loading } = usePlantImage(name);
  const [imgReady, setImgReady] = useState(false);
  const [failed, setFailed] = useState(false);

  const showSkeleton = loading || (url && !imgReady && !failed);
  const showFallback = !loading && (!url || failed);

  return (
    <div style={{ width: 40, height: 40, flexShrink: 0, position: 'relative' }}>
      {showSkeleton && <div style={{ width: 40, height: 40, borderRadius: 10, background: '#f3f4f6', position: 'absolute', inset: 0, animation: 'thumbShimmer 1.4s ease-in-out infinite' }} />}
      {showFallback && (
        <div style={{ width: 40, height: 40, borderRadius: 10, background: '#f3f4f6', border: '1.5px solid rgba(45,90,52,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <LeafIcon size={18} color="#3d8b3d" />
        </div>
      )}
      {url && !failed && (
        <img
          style={imgReady
            ? { width: 40, height: 40, borderRadius: 10, objectFit: 'cover', background: '#f3f4f6', border: '1.5px solid rgba(45,90,52,0.1)' }
            : { position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }
          }
          src={url}
          alt={name}
          loading="lazy"
          onLoad={() => setImgReady(true)}
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}

/* ─── Detail image (larger) ─── */
function DetailImage({ name }) {
  const { url, loading } = usePlantImage(name);
  const [imgReady, setImgReady] = useState(false);
  const [failed, setFailed] = useState(false);

  const fallback = (
    <div style={{ width: '100%', height: 200, background: 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <LeafIcon size={48} color="#3d8b3d" />
    </div>
  );

  if (loading || (url && !imgReady && !failed)) {
    return (
      <div style={{ width: '100%', height: 200, background: 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        {url && <img style={{ position: 'absolute', width: 0, height: 0, opacity: 0 }} src={url} alt={name} onLoad={() => setImgReady(true)} onError={() => setFailed(true)} />}
        <div style={{ width: 60, height: 60, borderRadius: '50%', background: 'rgba(255,255,255,0.5)', animation: 'thumbShimmer 1.4s ease-in-out infinite' }} />
      </div>
    );
  }

  if (!url || failed) return fallback;

  return <img style={{ width: '100%', height: 200, objectFit: 'cover', display: 'block' }} src={url} alt={name} />;
}

/* ─── Toxicity badge ─── */
const toxColors = {
  toxic: { bg: '#fde8e8', color: '#b91c1c', icon: '⚠' },
  mildlyToxic: { bg: '#fff7e6', color: '#b45309', icon: '⚠' },
  safe: { bg: '#e8f5e9', color: '#2e7d32', icon: '✓' },
};

function ToxBadge({ level, t }) {
  const c = toxColors[level] || toxColors.safe;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      fontFamily: "'Nunito', sans-serif", fontSize: '0.78rem', fontWeight: 700,
      color: c.color, background: c.bg, borderRadius: 8, padding: '3px 10px',
    }}>
      {c.icon} {t(`toxicity.${level}`)}
    </span>
  );
}

/* ─── Care info row ─── */
function CareRow({ icon, label, value }) {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
      <span style={{ lineHeight: 1.4, flexShrink: 0, marginTop: 1 }}>{icon}</span>
      <div style={{ minWidth: 0 }}>
        <p style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.72rem', fontWeight: 800, color: '#3d8b3d', textTransform: 'uppercase', letterSpacing: '0.04em', margin: 0 }}>
          {label}
        </p>
        <p style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.82rem', fontWeight: 600, color: '#374151', lineHeight: 1.4, margin: '2px 0 0' }}>
          {value}
        </p>
      </div>
    </div>
  );
}

/* ─── Detail panel (slides in from right inside the drawer) ─── */
function PlantDetailPanel({ plant, onBack, lang, t }) {
  const [fact, setFact] = useState(() => getRandomFact(plant.scientific));
  const care = getPlantCare(plant.scientific);
  const detailX = useMotionValue(0);

  const shuffleFact = useCallback(() => {
    setFact(getNextFact(plant.scientific));
  }, [plant.scientific]);

  const handleDetailDragEnd = useCallback((_, info) => {
    if (info.offset.x > 80 || info.velocity.x > 300) {
      animate(detailX, 400, { duration: 0.2 });
      onBack();
    } else {
      animate(detailX, 0, { type: 'spring', stiffness: 300, damping: 28 });
    }
  }, [onBack, detailX]);

  return (
    <motion.div
      style={{
        position: 'absolute', inset: 0, background: '#fff',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        zIndex: 2, x: detailX, touchAction: 'pan-y',
      }}
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={{ left: 0, right: 0.5 }}
      onDragEnd={handleDetailDragEnd}
      initial={{ x: '100%' }}
      animate={{ x: 0 }}
      exit={{ x: '100%' }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      {/* Back button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute', top: 12, left: 12, zIndex: 3,
          width: 32, height: 32, borderRadius: '50%',
          border: 'none', background: 'rgba(255,255,255,0.7)',
          backdropFilter: 'blur(8px)', boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: "'Nunito', sans-serif", fontSize: '1.5rem', fontWeight: 700, color: '#1a3a1a',
          lineHeight: 1, paddingBottom: 2,
          WebkitTapHighlightColor: 'transparent',
        }}
        aria-label="Back"
      >
        ‹
      </button>

      <DetailImage name={plant.scientific} />

      <div style={{ padding: '14px 20px 24px', display: 'flex', flexDirection: 'column', gap: 10, overflowY: 'auto', flex: 1 }}>
        <p style={{ fontFamily: "'Nunito', sans-serif", fontSize: '1.1rem', fontWeight: 800, color: '#1a3a1a', margin: 0 }}>
          {plant.scientific}
        </p>
        {plant.commonName[lang] && (
          <p style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.88rem', fontWeight: 600, color: '#6b7280', fontStyle: 'italic', margin: 0 }}>
            {plant.commonName[lang]}
          </p>
        )}

        {/* Toxicity */}
        {care?.toxicity && (
          <div style={{ margin: '2px 0' }}>
            <ToxBadge level={care.toxicity} t={t} />
          </div>
        )}

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(45,90,52,0.1)', margin: '2px 0' }} />

        {/* Care info */}
        {care && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <CareRow icon={<WaterDropIcon size={18} />} label={t('care.watering')} value={care.water[lang] || care.water.en} />
            <CareRow icon={<CalendarIcon size={18} />} label={t('care.frequency')} value={care.frequency[lang] || care.frequency.en} />
            <CareRow icon={<SunIcon size={18} />} label={t('care.sunlight')} value={care.sunlight[lang] || care.sunlight.en} />
          </div>
        )}

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(45,90,52,0.1)', margin: '2px 0' }} />

        {/* Fun fact */}
        {fact && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <p style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.72rem', fontWeight: 800, color: '#3d8b3d', textTransform: 'uppercase', letterSpacing: '0.04em', margin: 0, flex: 1 }}>
                {t('fact.label')}
              </p>
              <button
                onClick={shuffleFact}
                style={{
                  width: 28, height: 28, borderRadius: '50%', border: 'none',
                  background: 'rgba(61,139,61,0.1)', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  WebkitTapHighlightColor: 'transparent',
                  transition: 'background 0.2s',
                }}
                aria-label="Show another fact"
              >
                <RefreshIcon size={15} color="#3d8b3d" />
              </button>
            </div>
            <p style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.85rem', fontWeight: 600, color: '#374151', lineHeight: 1.5, margin: 0 }}>
              {fact[lang] || fact.en}
            </p>
          </>
        )}
      </div>
    </motion.div>
  );
}

/* ─── Main plant list – left side drawer ─── */
export default function PlantListModal({ open, onClose }) {
  const { lang, t } = useI18n();
  const plants = getPlantList();
  const [selectedPlant, setSelectedPlant] = useState(null);
  const { x, handleDragEnd } = useDragXToClose(onClose);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)',
            zIndex: 1000,
          }}
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, pointerEvents: 'none' }}
        >
          <motion.div
            style={{
              position: 'absolute', top: 0, left: 0, bottom: 0,
              width: '82vw', maxWidth: 340, background: '#fff',
              boxShadow: '4px 0 24px rgba(0,0,0,0.12)',
              display: 'flex', flexDirection: 'column',
              overflow: 'hidden', x,
            }}
            onClick={(e) => e.stopPropagation()}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            dragElastic={{ left: 0.5, right: 0 }}
            onDragEnd={handleDragEnd}
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '20px 18px 14px', borderBottom: '1px solid rgba(45,90,52,0.08)' }}>
              <h3 style={{ fontFamily: "'Nunito', sans-serif", fontSize: '1.1rem', fontWeight: 800, color: '#1a3a1a', margin: 0, flex: 1 }}>
                {t('plantList.title')}
              </h3>
              <span style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.75rem', fontWeight: 700, color: '#fff', background: '#3d8b3d', borderRadius: 10, padding: '2px 8px', lineHeight: 1.3 }}>
                {plants.length}
              </span>
            </div>

            {/* List */}
            <div
              style={{ margin: 0, padding: '8px 10px 16px', overflowY: 'auto', overscrollBehavior: 'contain', flex: 1, touchAction: 'pan-y' }}
            >
              {plants.map(({ scientific, commonName }, i) => (
                <div
                  key={scientific}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    padding: '8px 10px', borderRadius: 12, cursor: 'pointer',
                    WebkitTapHighlightColor: 'transparent',
                    background: i % 2 === 0 ? 'rgba(45,90,52,0.04)' : 'transparent',
                  }}
                  onClick={() => setSelectedPlant({ scientific, commonName })}
                >
                  <PlantThumb name={scientific} />
                  <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0, flex: 1 }}>
                    <span style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.92rem', fontWeight: 700, color: '#1a3a1a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {scientific}
                    </span>
                    {commonName[lang] && (
                      <span style={{ fontFamily: "'Nunito', sans-serif", fontSize: '0.8rem', fontWeight: 600, color: '#6b7280', fontStyle: 'italic', marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {commonName[lang]}
                      </span>
                    )}
                  </div>
                  <span style={{ fontFamily: "'Nunito', sans-serif", fontSize: '1.5rem', fontWeight: 700, color: '#b0b8c1', flexShrink: 0, marginRight: 2 }}>›</span>
                </div>
              ))}
            </div>

            {/* Detail panel slides over the list inside the same drawer */}
            <AnimatePresence>
              {selectedPlant && (
                <PlantDetailPanel
                  plant={selectedPlant}
                  onBack={() => setSelectedPlant(null)}
                  lang={lang}
                  t={t}
                />
              )}
            </AnimatePresence>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
