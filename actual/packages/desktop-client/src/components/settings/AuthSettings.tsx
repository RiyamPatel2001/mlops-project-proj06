import React from 'react';
import { Trans } from 'react-i18next';

import { Label } from '@actual-app/components/label';
import { Text } from '@actual-app/components/text';

import { Setting } from './UI';

export function AuthSettings() {
  return (
    <Setting
      primaryAction={
        <Label
          style={{ paddingTop: 5 }}
          title="Accounts now use Actual's built-in username/password registration and sign-in."
        />
      }
    >
      <Text>
        <Trans>
          <strong>Authentication</strong> is handled only through Actual
          accounts created on the sign-in screen.
        </Trans>
      </Text>
    </Setting>
  );
}
