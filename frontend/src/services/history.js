const STORAGE_KEY = 'florawatch_history';
const MAX_ENTRIES = 20;

export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch {
    return [];
  }
}

export function addToHistory(entry) {
  const history = getHistory();
  history.unshift({
    name: entry.name,
    commonName: entry.commonName || null,
    confidence: entry.confidence,
    timestamp: Date.now(),
  });
  if (history.length > MAX_ENTRIES) history.length = MAX_ENTRIES;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  return history;
}

export function clearHistory() {
  localStorage.removeItem(STORAGE_KEY);
}
