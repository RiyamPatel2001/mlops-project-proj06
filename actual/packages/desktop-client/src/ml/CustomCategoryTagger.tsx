import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@actual-app/components/button';
import { Input } from '@actual-app/components/input';
import { Text } from '@actual-app/components/text';
import { theme } from '@actual-app/components/theme';
import { View } from '@actual-app/components/view';

import { getCustomCategories, tagExample } from './mlService';

type Props = {
  accountId: string;
};

export function CustomCategoryTagger({ accountId: _accountId }: Props) {
  void _accountId;
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [payee, setPayee] = useState('');
  const [category, setCategory] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState('');
  const [customCategories, setCustomCategories] = useState<string[]>([]);

  const loadCategories = useCallback(async () => {
    const cats = await getCustomCategories();
    setCustomCategories(cats);
  }, []);

  useEffect(() => {
    void loadCategories();
  }, [loadCategories]);

  async function handleSubmit() {
    if (!payee.trim() || !category.trim()) return;
    setSubmitting(true);
    setMessage('');
    const result = await tagExample({
      payee: payee.trim(),
      custom_category: category.trim(),
    });
    setSubmitting(false);
    if (result.ok) {
      setMessage(t('Example saved'));
      setPayee('');
      setCategory('');
      void loadCategories();
    } else {
      if (result.reason === 'auth') {
        setMessage(t('Sign in to Actual to save examples'));
      } else if (result.reason === 'network') {
        setMessage(t('Failed to reach the ML service'));
      } else {
        setMessage(t('Failed to save example'));
      }
    }
  }

  if (!open) {
    return (
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
        <Button variant="bare" onPress={() => setOpen(true)}>
          {t('Tag Custom Category')}
        </Button>
        {customCategories.length > 0 && (
          <Text
            style={{
              fontSize: '0.8em',
              color: theme.pageTextSubdued,
            }}
          >
            ({customCategories.length} {t('custom')})
          </Text>
        )}
      </View>
    );
  }

  return (
    <View
      style={{
        padding: 10,
        border: `1px solid ${theme.tableBorder}`,
        borderRadius: 6,
        backgroundColor: theme.tableBackground,
        gap: 8,
        minWidth: 280,
      }}
    >
      <View
        style={{
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Text style={{ fontWeight: 600 }}>{t('Tag Custom Category')}</Text>
        <Button variant="bare" onPress={() => setOpen(false)}>
          ✕
        </Button>
      </View>

      <Text style={{ fontSize: '0.85em', color: theme.pageTextSubdued }}>
        {t('Teach the model your personal categories by providing examples.')}
      </Text>

      <View style={{ gap: 4 }}>
        <Input
          placeholder={t('Payee (e.g. WHOLE FOODS MKT)')}
          value={payee}
          onChangeValue={setPayee}
        />
        <Input
          placeholder={t('Category name')}
          value={category}
          onChangeValue={setCategory}
        />
      </View>

      <View style={{ flexDirection: 'row', gap: 6, alignItems: 'center' }}>
        <Button
          variant="primary"
          isDisabled={!payee.trim() || !category.trim() || submitting}
          onPress={handleSubmit}
        >
          {submitting ? t('Saving…') : t('Save Example')}
        </Button>
        {message && (
          <Text style={{ fontSize: '0.85em', color: theme.pageTextSubdued }}>
            {message}
          </Text>
        )}
      </View>

      {customCategories.length > 0 && (
        <View style={{ gap: 2 }}>
          <Text
            style={{
              fontSize: '0.8em',
              fontWeight: 600,
              color: theme.pageTextSubdued,
            }}
          >
            {t('Your custom categories:')}
          </Text>
          <View
            style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 4 }}
          >
            {customCategories.map(cat => (
              <Text
                key={cat}
                style={{
                  fontSize: '0.8em',
                  padding: '2px 6px',
                  borderRadius: 3,
                  backgroundColor: theme.tableRowHeaderBackground,
                  color: theme.tableText,
                }}
              >
                {cat}
              </Text>
            ))}
          </View>
        </View>
      )}
    </View>
  );
}
