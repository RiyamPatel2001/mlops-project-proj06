import React from 'react';
import { Trans, useTranslation } from 'react-i18next';

import { Button } from '@actual-app/components/button';
import { Label } from '@actual-app/components/label';
import { Text } from '@actual-app/components/text';
import { theme } from '@actual-app/components/theme';
import { View } from '@actual-app/components/view';

import { useLoginMethod, useMultiuserEnabled } from '#components/ServerContext';
import { useSyncServerStatus } from '#hooks/useSyncServerStatus';
import { pushModal } from '#modals/modalsSlice';
import { useDispatch } from '#redux';

import { Setting } from './UI';

export function AuthSettings() {
  const { t } = useTranslation();

  const multiuserEnabled = useMultiuserEnabled();
  const loginMethod = useLoginMethod();
  const dispatch = useDispatch();
  const serverStatus = useSyncServerStatus();

  // Hide the OpenID block entirely when no server is configured
  if (serverStatus === 'no-server') {
    return null;
  }

  const isOffline = serverStatus === 'offline';

  return (
    <Setting
      primaryAction={
        <>
          <label>
            <Trans>OpenID is</Trans>{' '}
            <label style={{ fontWeight: 'bold' }}>
              {loginMethod === 'openid' ? t('enabled') : t('disabled')}
            </label>
          </label>
          {isOffline && (
            <View>
              <Text style={{ paddingTop: 5, color: theme.warningText }}>
                <Trans>
                  Server is offline. OpenID settings are unavailable.
                </Trans>
              </Text>
            </View>
          )}
          {loginMethod === 'password' && (
            <>
              <Button
                id="start-using"
                style={{
                  marginTop: '10px',
                }}
                variant="normal"
                isDisabled={isOffline}
                onPress={() =>
                  dispatch(
                    pushModal({
                      modal: {
                        name: 'enable-openid',
                        options: {},
                      },
                    }),
                  )
                }
              >
                <Trans>Start using OpenID</Trans>
              </Button>
              <Label
                style={{ paddingTop: 5 }}
                title={t(
                  'OpenID is optional when you want sign-in to be handled by an external identity provider.',
                )}
              />
            </>
          )}
          {loginMethod !== 'password' && (
            <>
              <Button
                style={{
                  marginTop: '10px',
                }}
                variant="normal"
                isDisabled={isOffline}
                onPress={() =>
                  dispatch(
                    pushModal({
                      modal: {
                        name: 'enable-password-auth',
                        options: {},
                      },
                    }),
                  )
                }
              >
                <Trans>Disable OpenID</Trans>
              </Button>
              {multiuserEnabled && (
                <Text style={{ paddingTop: 5, color: theme.errorText }}>
                  <Trans>
                    Disabling OpenID will switch the server back to local
                    username/password sign-in.
                  </Trans>
                </Text>
              )}
            </>
          )}
        </>
      }
    >
      <Text>
        <Trans>
          <strong>Authentication method</strong> modifies how users log in to
          the system.
        </Trans>
      </Text>
    </Setting>
  );
}
