import { useCallback, useRef, useState } from 'react';

import {
  classifyTransaction,
  type ClassifyResponse,
} from './mlService';

export type MLPrediction = {
  category: string;
  confidence: number;
  source: string;
  model_version: string;
};

/**
 * Hook that classifies a batch of imported transactions against the ML
 * serving endpoint.  Returns a map of trx_id → prediction.
 */
export function useMLClassify(accountId: string) {
  const [predictions, setPredictions] = useState<
    Record<string, MLPrediction>
  >({});
  const [isClassifying, setIsClassifying] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const classifyBatch = useCallback(
    async (
      transactions: Array<{
        trx_id: string;
        payee_name?: string;
        date?: string;
        amount?: number;
        imported_id?: string;
        category?: string;
      }>,
    ) => {
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      setIsClassifying(true);
      const newPreds: Record<string, MLPrediction> = {};

      const toClassify = transactions.filter(
        t => !t.category && t.payee_name && t.date,
      );

      const results = await Promise.allSettled(
        toClassify.map(t =>
          classifyTransaction({
            account: accountId,
            date: t.date!,
            amount: t.amount ?? 0,
            payee_name: t.payee_name!,
            imported_id: t.imported_id ?? t.trx_id,
          }),
        ),
      );

      results.forEach((result, idx) => {
        if (result.status === 'fulfilled' && result.value) {
          const t = toClassify[idx];
          newPreds[t.trx_id] = {
            category: result.value.category,
            confidence: result.value.confidence,
            source: result.value.source,
            model_version: result.value.model_version,
          };
        }
      });

      setPredictions(prev => ({ ...prev, ...newPreds }));
      setIsClassifying(false);
      return newPreds;
    },
    [accountId],
  );

  const clearPredictions = useCallback(() => setPredictions({}), []);

  return { predictions, isClassifying, classifyBatch, clearPredictions };
}
