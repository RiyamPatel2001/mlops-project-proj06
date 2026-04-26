// @ts-strict-ignore
import React, { useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';

import { Button } from '@actual-app/components/button';
import { useResponsive } from '@actual-app/components/hooks/useResponsive';
import { BigInput } from '@actual-app/components/input';
import { styles } from '@actual-app/components/styles';
import { Text } from '@actual-app/components/text';
import { theme } from '@actual-app/components/theme';
import { View } from '@actual-app/components/view';
import { send } from '@actual-app/core/platform/client/connection';

import { useDispatch } from '#redux';
import { loggedIn } from '#users/usersSlice';

import { Title, useBootstrapped } from './common';

function AuthForm({
  mode,
  setError,
}: {
  mode: 'login' | 'register';
  setError: (error: string | null) => void;
}) {
  const dispatch = useDispatch();
  const { t } = useTranslation();
  const { isNarrowWidth } = useResponsive();
  const [userName, setUserName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit() {
    if (submitting) {
      return;
    }

    if (userName.trim() === '' || password === '') {
      setError(userName.trim() === '' ? 'invalid-username' : 'invalid-password');
      return;
    }

    setError(null);
    setSubmitting(true);
    const result =
      mode === 'register'
        ? await send('subscribe-register', {
            userName,
            displayName,
            password,
          })
        : await send('subscribe-sign-in', {
            userName,
            password,
            loginMethod: 'password',
          });
    setSubmitting(false);

    if (result?.error) {
      setError(result.error);
      return;
    }

    dispatch(loggedIn());
  }

  return (
    <View style={{ flexDirection: 'column', marginTop: 5, gap: '1rem' }}>
      <BigInput
        autoFocus
        placeholder={t('Username')}
        onChangeValue={setUserName}
        style={{ width: '100%' }}
        onEnter={onSubmit}
      />

      {mode === 'register' && (
        <BigInput
          placeholder={t('Display name (optional)')}
          onChangeValue={setDisplayName}
          style={{ width: '100%' }}
          onEnter={onSubmit}
        />
      )}

      <View
        style={{
          flexDirection: isNarrowWidth ? 'column' : 'row',
          gap: '1rem',
        }}
      >
        <BigInput
          placeholder={t('Password')}
          type="password"
          onChangeValue={setPassword}
          style={{ flex: 1 }}
          onEnter={onSubmit}
        />
        <Button
          variant="primary"
          style={{
            fontSize: 15,
            width: isNarrowWidth ? '100%' : 170,
            ...(isNarrowWidth ? { padding: 10 } : null),
          }}
          isDisabled={submitting}
          onPress={onSubmit}
        >
          {mode === 'register' ? <Trans>Create account</Trans> : <Trans>Sign in</Trans>}
        </Button>
      </View>
    </View>
  );
}

export function Login() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [error, setError] = useState<string | null>(null);
  const { checked } = useBootstrapped();

  function getErrorMessage(currentError) {
    switch (currentError) {
      case 'invalid-username':
        return t('Username cannot be empty');
      case 'invalid-password':
        return t('Invalid username or password');
      case 'user-already-exists':
        return t('That username is already in use');
      case 'network-failure':
        return t('Unable to contact the server');
      case 'internal-error':
        return t('Internal error');
      default:
        return t(`An unknown error occurred: {{error}}`, {
          error: currentError,
        });
    }
  }

  if (!checked) {
    return null;
  }

  return (
    <View style={{ maxWidth: 450, marginTop: -30, color: theme.pageText }}>
      <Title
        text={
          mode === 'register'
            ? t('Create your Actual account')
            : t('Sign in to Actual')
        }
      />

      <Text
        style={{
          fontSize: 16,
          color: theme.pageTextDark,
          lineHeight: 1.4,
          marginBottom: 10,
        }}
      >
        {mode === 'register' ? (
          <Trans>Create a username and password for this Actual server.</Trans>
        ) : (
          <Trans>Sign in with the username and password for your Actual account.</Trans>
        )}
      </Text>

      <AuthForm mode={mode} setError={setError} />

      <View style={{ marginTop: 10, alignItems: 'flex-end' }}>
        <Button
          variant="bare"
          style={{
            ...styles.verySmallText,
            color: theme.pageTextLight,
            paddingTop: 5,
            width: 'fit-content',
          }}
          onPress={() => {
            setError(null);
            setMode(mode === 'login' ? 'register' : 'login');
          }}
        >
          {mode === 'login' ? (
            <Trans>Need an account? Create one</Trans>
          ) : (
            <Trans>Already have an account? Sign in</Trans>
          )}
        </Button>
      </View>

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
    </View>
  );
}
