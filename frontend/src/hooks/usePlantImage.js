import { useState, useEffect } from 'react';

const cache = {};

/**
 * Fetches a thumbnail from Wikipedia for a given plant name.
 * Returns { url, loading }.
 */
export default function usePlantImage(name) {
  const [url, setUrl] = useState(cache[name] || null);
  const [loading, setLoading] = useState(!cache[name]);

  useEffect(() => {
    if (!name) return;
    if (cache[name]) {
      setUrl(cache[name]);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);

    const endpoint = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(name)}`;

    fetch(endpoint)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled) return;
        const src = data?.thumbnail?.source || null;
        if (src) cache[name] = src;
        setUrl(src);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [name]);

  return { url, loading };
}
