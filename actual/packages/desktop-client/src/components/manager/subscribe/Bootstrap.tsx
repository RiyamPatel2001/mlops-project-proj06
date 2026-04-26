// @ts-strict-ignore
import React, { useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';

import { Button } from '@actual-app/components/button';
import { useResponsive } from '@actual-app/components/hooks/useResponsive';
import { BigInput } from '@actual-app/components/input';
import { Paragraph } from '@actual-app/components/paragraph';
import { Text } from '@actual-app/components/text';
import { theme } from '@actual-app/components/theme';
import { View } from '@actual-app/components/view';
import { send } from '@actual-app/core/platform/client/connection';

import { createBudget } from '#budgetfiles/budgetfilesSlice';
import { useNavigate } from '#hooks/useNavigate';
import { useDispatch } from '#redux';
import { loggedIn } from '#users/usersSlice';

import { Title, useBootstrapped } from './common';

export function Bootstrap() {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { isNarrowWidth } = useResponsive();
  const { checked } = useBootstrapped();
  const [userName, setUserName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function getErrorMessage(currentError) {
    switch (currentError) {
      case 'invalid-username':
        return t('Username cannot be empty');
      case 'invalid-password':
        return t('Password cannot be empty');
      case 'password-match':
        return t('Passwords do not match');
      case 'user-already-exists':
        return t('That username is already in use');
      case 'network-failure':
        return t('Unable to contact the server');
      default:
        return t(`An unknown error occurred: {{error}}`, {
          error: currentError,
        });
    }
  }

  async function onSubmit() {
    if (submitting) {
      return;
    }

    if (userName.trim() === '') {
      setError('invalid-username');
      return;
    }

    if (password === '') {
      setError('invalid-password');
      return;
    }

    if (password !== confirmPassword) {
      setError('password-match');
      return;
    }

    setError(null);
    setSubmitting(true);
    const { error } = await send('subscribe-bootstrap', {
      userName,
      displayName,
      password,
    });
    setSubmitting(false);

    if (error) {
      setError(error);
      return;
    }

    dispatch(loggedIn());
    void navigate('/');
  }

  async function onDemo() {
    await dispatch(createBudget({ demoMode: true }));
  }

  if (!checked) {
    return null;
  }

  return (
    <View style={{ maxWidth: 450 }}>
      <Title text={t('Create your first Actual account')} />
      <Paragraph style={{ fontSize: 16, color: theme.pageTextDark }}>
        <Trans>
          This server does not have any users yet. Create the first account to
          finish setup and start using Actual.
        </Trans>
      </Paragraph>

      {error && (
        <Text
          style={{
            marginTop: 20,
            color: theme.errorText,
            borderRadius: 4,
            fontSize: 15,
          }}
        >
          {getErrorMessage(error)}
        </Text>
      )}

      <View style={{ flexDirection: 'column', gap: '1rem', marginTop: 10 }}>
        <BigInput
          autoFocus
          placeholder={t('Username')}
          onChangeValue={setUserName}
          onEnter={onSubmit}
        />
        <BigInput
          placeholder={t('Display name (optional)')}
          onChangeValue={setDisplayName}
          onEnter={onSubmit}
        />
        <BigInput
          placeholder={t('Password')}
          type="password"
          onChangeValue={setPassword}
          onEnter={onSubmit}
        />
        <BigInput
          placeholder={t('Confirm password')}
          type="password"
          onChangeValue={setConfirmPassword}
          onEnter={onSubmit}
        />
        <View
          style={{
            flexDirection: isNarrowWidth ? 'column' : 'row',
            justifyContent: 'space-between',
            gap: '1rem',
          }}
        >
          <Button
            variant="bare"
            style={{ fontSize: 15, color: theme.pageTextLink }}
            onPress={onDemo}
          >
            {t('Try Demo')}
          </Button>
          <Button
            variant="primary"
            style={{ fontSize: 15, width: isNarrowWidth ? '100%' : 190 }}
            isDisabled={submitting}
            onPress={onSubmit}
          >
            <Trans>Create account</Trans>
          </Button>
        </View>
      </View>
    </View>
  );
}
