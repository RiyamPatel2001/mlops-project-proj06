// @ts-strict-ignore
import React, { useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';

import { ButtonWithLoading } from '@actual-app/components/button';
import { useResponsive } from '@actual-app/components/hooks/useResponsive';
import { BigInput } from '@actual-app/components/input';
import { styles } from '@actual-app/components/styles';
import { Text } from '@actual-app/components/text';
import { theme } from '@actual-app/components/theme';
import { View } from '@actual-app/components/view';

import { useNavigate } from '#hooks/useNavigate';
import { useDispatch } from '#redux';
import { loggedIn } from '#users/usersSlice';

import {
  getStoredMLUsername,
  registerMLUser,
  signInMLUser,
} from '../../../ml/mlService';

type AuthMode = 'register' | 'login';

function AuthModeButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        border: `1px solid ${active ? theme.buttonPrimaryBorder : theme.pillBorder}`,
        borderRadius: 999,
        padding: '8px 14px',
        cursor: 'pointer',
        fontSize: 15,
        fontWeight: 600,
        color: active ? theme.buttonPrimaryText : theme.pageText,
        background: active
          ? theme.buttonPrimaryBackground
          : theme.pillBackground,
      }}
    >
      {children}
    </button>
  );
}

export function MLServiceAuth() {
  const { t } = useTranslation();
  const { isNarrowWidth } = useResponsive();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [mode, setMode] = useState<AuthMode>('register');
  const [username, setUsername] = useState(() => getStoredMLUsername());
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageTone, setMessageTone] = useState<'notice' | 'error'>('notice');

  async function onSubmit() {
    const trimmedUsername = username.trim();
    if (loading) {
      return;
    }

    if (!trimmedUsername || !password) {
      setMessageTone('error');
      setMessage(t('Username and password are required.'));
      return;
    }

    setLoading(true);
    if (mode === 'register') {
      const registration = await registerMLUser(trimmedUsername, password);
      setLoading(false);

      if (registration.ok) {
        setMode('login');
        setPassword('');
        setMessageTone('notice');
        setMessage(
          t(
            'Account created. Sign in with your new username and password to continue.',
          ),
        );
      } else {
        setMessageTone('error');
        setMessage(t(registration.message));
      }
      return;
    }

    const login = await signInMLUser(trimmedUsername, password);
    setLoading(false);

    if (login.ok) {
      setPassword('');
      await dispatch(loggedIn());
      void navigate('/', { replace: true });
    } else {
      setMessageTone('error');
      setMessage(t(login.message));
    }
  }

  return (
    <View
      style={{
        marginTop: 26,
        paddingTop: 22,
        borderTop: `1px solid ${theme.cardBorder}`,
        gap: '0.85rem',
      }}
    >
      <Text
        style={{
          fontSize: 22,
          fontWeight: 700,
          color: theme.pageText,
        }}
      >
        <Trans>Sign in to Actual Budget</Trans>
      </Text>
      <Text
        style={{
          fontSize: 15,
          color: theme.pageTextDark,
          lineHeight: 1.5,
        }}
      >
        <Trans>
          Sign in with your account to open your budgets and continue to the
          Actual Budget app.
        </Trans>
      </Text>

      <View style={{ flexDirection: 'row', gap: '0.75rem', marginTop: 4 }}>
        <AuthModeButton
          active={mode === 'register'}
          onClick={() => {
            setMode('register');
            setMessage('');
          }}
        >
          <Trans>New user</Trans>
        </AuthModeButton>
        <AuthModeButton
          active={mode === 'login'}
          onClick={() => {
            setMode('login');
            setMessage('');
          }}
        >
          <Trans>Returning user</Trans>
        </AuthModeButton>
      </View>

      <Text
        style={{
          ...styles.verySmallText,
          color: theme.pageTextLight,
        }}
      >
        {mode === 'register' ? (
          <Trans>
            Create your Actual Budget account with a unique username and
            password.
          </Trans>
        ) : (
          <Trans>Use your existing account credentials.</Trans>
        )}
      </Text>

      <View
        style={{
          flexDirection: 'column',
          gap: '0.85rem',
        }}
      >
        <BigInput
          placeholder={t('Username')}
          value={username}
          onChangeValue={setUsername}
          style={{ width: '100%' }}
          onEnter={onSubmit}
        />
        <View
          style={{
            flexDirection: isNarrowWidth ? 'column' : 'row',
            gap: '1rem',
          }}
        >
          <BigInput
            placeholder={t('Password')}
            type="password"
            value={password}
            onChangeValue={setPassword}
            style={{ flex: 1 }}
            onEnter={onSubmit}
          />
          <ButtonWithLoading
            variant="primary"
            isLoading={loading}
            style={{
              fontSize: 15,
              width: isNarrowWidth ? '100%' : 170,
              ...(isNarrowWidth ? { padding: 10 } : null),
            }}
            onPress={onSubmit}
          >
            {mode === 'register' ? (
              <Trans>Create account</Trans>
            ) : (
              <Trans>Sign in</Trans>
            )}
          </ButtonWithLoading>
        </View>
      </View>

      {message ? (
        <Text
          style={{
            fontSize: 14,
            lineHeight: 1.45,
            color:
              messageTone === 'notice' ? theme.noticeText : theme.errorText,
          }}
        >
          {message}
        </Text>
      ) : null}
    </View>
  );
}
