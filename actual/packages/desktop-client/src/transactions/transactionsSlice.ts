import type { TransactionEntity } from '@actual-app/core/types/models';
import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

import { resetApp } from '#app/appSlice';

const sliceName = 'transactions';

type TransactionsState = {
  newTransactions: Array<TransactionEntity['id']>;
  matchedTransactions: Array<TransactionEntity['id']>;
  lastTransaction: TransactionEntity | null;
  mlCategoryPredictions: Record<
    TransactionEntity['id'],
    {
      transactionId: TransactionEntity['id'];
      payee: string;
      amount: number;
      date: string;
      predictedCategory: string;
      confidence: number | null;
      source: string;
      flashVersion: number;
    }
  >;
};

const initialState: TransactionsState = {
  newTransactions: [],
  matchedTransactions: [],
  lastTransaction: null,
  mlCategoryPredictions: {},
};

type SetNewTransactionsPayload = {
  newTransactions: Array<TransactionEntity['id']>;
  matchedTransactions: Array<TransactionEntity['id']>;
};

type UpdateNewTransactionsPayload = {
  id: TransactionEntity['id'];
};

type SetLastTransactionPayload = {
  transaction: TransactionEntity;
};

type RegisterMlCategoryPredictionPayload = {
  transactionId: TransactionEntity['id'];
  payee: string;
  amount: number;
  date: string;
  predictedCategory: string;
  confidence: number | null;
  source: string;
};

type ClearMlCategoryPredictionPayload = {
  transactionId: TransactionEntity['id'];
};

const transactionsSlice = createSlice({
  name: sliceName,
  initialState,
  reducers: {
    setNewTransactions(
      state,
      action: PayloadAction<SetNewTransactionsPayload>,
    ) {
      state.newTransactions = action.payload.newTransactions
        ? [...state.newTransactions, ...action.payload.newTransactions]
        : state.newTransactions;

      state.matchedTransactions = action.payload.matchedTransactions
        ? [...state.matchedTransactions, ...action.payload.matchedTransactions]
        : state.matchedTransactions;
    },
    updateNewTransactions(
      state,
      action: PayloadAction<UpdateNewTransactionsPayload>,
    ) {
      state.newTransactions = state.newTransactions.filter(
        id => id !== action.payload.id,
      );
      state.matchedTransactions = state.matchedTransactions.filter(
        id => id !== action.payload.id,
      );
    },
    setLastTransaction(
      state,
      action: PayloadAction<SetLastTransactionPayload>,
    ) {
      state.lastTransaction = action.payload.transaction;
    },
    registerMlCategoryPrediction(
      state,
      action: PayloadAction<RegisterMlCategoryPredictionPayload>,
    ) {
      const existing = state.mlCategoryPredictions[action.payload.transactionId];
      state.mlCategoryPredictions[action.payload.transactionId] = {
        ...action.payload,
        flashVersion: (existing?.flashVersion ?? 0) + 1,
      };
    },
    clearMlCategoryPrediction(
      state,
      action: PayloadAction<ClearMlCategoryPredictionPayload>,
    ) {
      delete state.mlCategoryPredictions[action.payload.transactionId];
    },
  },
  extraReducers: builder => {
    builder.addCase(resetApp, () => initialState);
  },
});

export const { name, reducer, getInitialState } = transactionsSlice;
export const actions = {
  ...transactionsSlice.actions,
};

export const {
  setNewTransactions,
  updateNewTransactions,
  setLastTransaction,
  registerMlCategoryPrediction,
  clearMlCategoryPrediction,
} = transactionsSlice.actions;
