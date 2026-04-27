import { beforeEach, describe, expect, it, vi } from 'vitest';

import { clearMlCategoryPrediction } from '#transactions/transactionsSlice';

import * as mlService from './mlService';
import {
  shouldSyncMlCategoryFeedbackOnCategorySave,
  syncMlCategoryFeedbackOnEdit,
} from './transactionCategory';

describe('transactionCategory', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('records reviewed feedback when the user keeps the predicted category', async () => {
    const submitFeedback = vi
      .spyOn(mlService, 'submitFeedback')
      .mockResolvedValue(true);
    const dispatch = vi.fn();

    const result = await syncMlCategoryFeedbackOnEdit({
      dispatch,
      state: {
        transactions: {
          mlCategoryPredictions: {
            'txn-1': {
              transactionId: 'txn-1',
              payee: 'WHOLE FOODS',
              amount: -4215,
              date: '2026-04-27',
              predictedCategory: 'Groceries',
              confidence: 0.93,
              source: 'layer1',
              flashVersion: 1,
            },
          },
        },
      } as never,
      transaction: { id: 'txn-1' } as never,
      categoryGroups: [
        {
          id: 'group-1',
          name: 'Everyday',
          categories: [{ id: 'cat-1', name: 'Groceries' }],
        },
      ] as never,
      nextCategoryId: 'cat-1',
    });

    expect(result).toBe(true);
    expect(submitFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        transaction_id: 'txn-1',
        original_prediction: 'Groceries',
        final_label: 'Groceries',
        reviewed_by_user: true,
      }),
    );
    expect(dispatch).toHaveBeenCalledWith(
      clearMlCategoryPrediction({ transactionId: 'txn-1' }),
    );
  });

  it('records the corrected final label when the user changes the prediction', async () => {
    const submitFeedback = vi
      .spyOn(mlService, 'submitFeedback')
      .mockResolvedValue(true);

    await syncMlCategoryFeedbackOnEdit({
      dispatch: vi.fn(),
      state: {
        transactions: {
          mlCategoryPredictions: {
            'txn-2': {
              transactionId: 'txn-2',
              payee: 'Naya',
              amount: -1800,
              date: '2026-04-27',
              predictedCategory: 'Clothing',
              confidence: 0.803,
              source: 'layer1',
              flashVersion: 1,
            },
          },
        },
      } as never,
      transaction: { id: 'txn-2' } as never,
      categoryGroups: [
        {
          id: 'group-1',
          name: 'Everyday',
          categories: [{ id: 'cat-2', name: 'General' }],
        },
      ] as never,
      nextCategoryId: 'cat-2',
    });

    expect(submitFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        transaction_id: 'txn-2',
        original_prediction: 'Clothing',
        final_label: 'General',
        reviewed_by_user: true,
      }),
    );
  });

  it('syncs ML feedback only for explicit category saves on predicted transactions', () => {
    const state = {
      transactions: {
        mlCategoryPredictions: {
          'txn-1': {
            transactionId: 'txn-1',
            payee: 'WHOLE FOODS',
            amount: -4215,
            date: '2026-04-27',
            predictedCategory: 'Groceries',
            confidence: 0.93,
            source: 'layer1',
            flashVersion: 1,
          },
        },
      },
    } as never;

    expect(
      shouldSyncMlCategoryFeedbackOnCategorySave({
        updatedFieldName: 'category',
        state,
        transactionId: 'txn-1',
      }),
    ).toBe(true);

    expect(
      shouldSyncMlCategoryFeedbackOnCategorySave({
        updatedFieldName: 'notes',
        state,
        transactionId: 'txn-1',
      }),
    ).toBe(false);

    expect(
      shouldSyncMlCategoryFeedbackOnCategorySave({
        updatedFieldName: 'category',
        state,
        transactionId: 'txn-missing',
      }),
    ).toBe(false);
  });
});
