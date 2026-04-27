import { send } from '@actual-app/core/platform/client/connection';
import type {
  AccountEntity,
  CategoryEntity,
  CategoryGroupEntity,
  PayeeEntity,
  TransactionEntity,
} from '@actual-app/core/types/models';

import type { AppDispatch, RootState } from '#redux/store';
import {
  clearMlCategoryPrediction,
  registerMlCategoryPrediction,
} from '#transactions/transactionsSlice';

import { classifyTransaction, submitFeedback } from './mlService';

function getCategoryNameMap(categoryGroups: CategoryGroupEntity[]) {
  const map: Record<string, string> = {};
  categoryGroups.forEach(group => {
    group.categories?.forEach(category => {
      map[normalizeCategoryName(category.name)] = category.id;
    });
  });
  return map;
}

function getCategoryIdMap(categoryGroups: CategoryGroupEntity[]) {
  const map: Record<string, CategoryEntity> = {};
  categoryGroups.forEach(group => {
    group.categories?.forEach(category => {
      map[category.id] = category;
    });
  });
  return map;
}

export function normalizeCategoryName(category: string | null | undefined) {
  return category?.trim() ?? '';
}

export function getCategoryNameById(
  categoryGroups: CategoryGroupEntity[],
  categoryId: string | null | undefined,
) {
  if (!categoryId) {
    return '';
  }
  return normalizeCategoryName(getCategoryIdMap(categoryGroups)[categoryId]?.name);
}

export async function ensureCategoryIds(
  categoryGroups: CategoryGroupEntity[],
  categoryNames: string[],
) {
  const categoryIdsByName = getCategoryNameMap(categoryGroups);
  const missingNames = [...new Set(categoryNames.map(normalizeCategoryName))].filter(
    name => name && !categoryIdsByName[name],
  );

  if (missingNames.length === 0) {
    return categoryIdsByName;
  }

  let predictionGroupId = categoryGroups.find(
    group => group.name === 'ML Predictions',
  )?.id;

  if (!predictionGroupId) {
    try {
      predictionGroupId = await send('category-group-create', {
        name: 'ML Predictions',
      });
    } catch {
      predictionGroupId = undefined;
    }
  }

  for (const name of missingNames) {
    try {
      const createdId = await send('category-create', {
        name,
        groupId: predictionGroupId,
      });
      if (createdId) {
        categoryIdsByName[name] = createdId;
      }
    } catch {
      // If category creation fails, let the caller treat the prediction as best-effort.
    }
  }

  return categoryIdsByName;
}

function resolvePayeeName(
  transaction: TransactionEntity,
  payees: PayeeEntity[],
) {
  if (!transaction.payee) {
    return '';
  }

  const payee = payees.find(item => item.id === transaction.payee);
  return payee?.name?.trim() ?? '';
}

function isTransferPayee(transaction: TransactionEntity, payees: PayeeEntity[]) {
  if (!transaction.payee) {
    return false;
  }

  const payee = payees.find(item => item.id === transaction.payee);
  return !!payee?.transfer_acct;
}

function shouldPredictCategory(
  transaction: TransactionEntity,
  accounts: AccountEntity[],
  payees: PayeeEntity[],
) {
  if (transaction.category || transaction.is_parent || transaction.is_child) {
    return false;
  }

  if (!transaction.id || transaction.id.startsWith('temp')) {
    return false;
  }

  if (!transaction.date || transaction.amount == null) {
    return false;
  }

  const payeeName = resolvePayeeName(transaction, payees);
  if (!payeeName || isTransferPayee(transaction, payees)) {
    return false;
  }

  const account = accounts.find(item => item.id === transaction.account);
  if (account?.offbudget === 1) {
    return false;
  }

  return true;
}

export async function autoCategorizeTransaction({
  transaction,
  accounts,
  payees,
  categoryGroups,
  dispatch,
  onCategoryApplied,
}: {
  transaction: TransactionEntity;
  accounts: AccountEntity[];
  payees: PayeeEntity[];
  categoryGroups: CategoryGroupEntity[];
  dispatch: AppDispatch;
  onCategoryApplied?: (categoryId: string, predictedCategory: string) => void;
}) {
  if (!shouldPredictCategory(transaction, accounts, payees)) {
    return false;
  }

  const payee = resolvePayeeName(transaction, payees);
  const prediction = await classifyTransaction({
    account: transaction.account ?? '',
    date: transaction.date,
    amount: transaction.amount,
    payee_name: payee,
    imported_id: transaction.id,
    requestMode: 'interactive',
  });

  const predictedCategory = normalizeCategoryName(prediction?.category);
  if (!prediction || !predictedCategory) {
    return false;
  }

  const categoryIdsByName = await ensureCategoryIds(categoryGroups, [
    predictedCategory,
  ]);
  const categoryId = categoryIdsByName[predictedCategory];

  if (!categoryId) {
    return false;
  }

  await send('api/transaction-update', {
    id: transaction.id,
    fields: { category: categoryId },
  });

  dispatch(
    registerMlCategoryPrediction({
      transactionId: transaction.id,
      payee,
      amount: transaction.amount,
      date: transaction.date,
      predictedCategory,
      confidence: prediction.confidence ?? null,
      source: prediction.source,
    }),
  );

  onCategoryApplied?.(categoryId, predictedCategory);

  void submitFeedback({
    transaction_id: transaction.id,
    payee,
    amount: transaction.amount,
    date: transaction.date,
    original_prediction: predictedCategory,
    original_confidence: prediction.confidence ?? null,
    source: prediction.source,
    final_label: predictedCategory,
    reviewed_by_user: false,
    timestamp: new Date().toISOString(),
  });

  return true;
}

export function getMlCategoryPrediction(
  state: RootState,
  transactionId: TransactionEntity['id'],
) {
  return state.transactions.mlCategoryPredictions[transactionId] ?? null;
}

export async function syncMlCategoryFeedbackOnEdit({
  dispatch,
  state,
  transaction,
  categoryGroups,
  nextCategoryId,
}: {
  dispatch: AppDispatch;
  state: RootState;
  transaction: TransactionEntity;
  categoryGroups: CategoryGroupEntity[];
  nextCategoryId: string | null | undefined;
}) {
  const prediction = getMlCategoryPrediction(state, transaction.id);
  if (!prediction) {
    return false;
  }

  const finalLabel = getCategoryNameById(categoryGroups, nextCategoryId);
  if (normalizeCategoryName(finalLabel) === prediction.predictedCategory) {
    return false;
  }

  await submitFeedback({
    transaction_id: transaction.id,
    payee: prediction.payee,
    amount: prediction.amount,
    date: prediction.date,
    original_prediction: prediction.predictedCategory,
    original_confidence: prediction.confidence,
    source: prediction.source,
    final_label: finalLabel,
    reviewed_by_user: true,
    timestamp: new Date().toISOString(),
  });

  dispatch(clearMlCategoryPrediction({ transactionId: transaction.id }));
  return true;
}
