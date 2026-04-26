// @ts-strict-ignore
import React from 'react';

import { theme } from '@actual-app/components/theme';
import { View } from '@actual-app/components/view';

import { useBootstrapped } from './common';
import { MLServiceAuth } from './MLServiceAuth';

export function Login() {
  const { checked } = useBootstrapped();

  if (!checked) {
    return null;
  }

  return (
    <View style={{ maxWidth: 450, marginTop: -30, color: theme.pageText }}>
      <MLServiceAuth />
    </View>
  );
}
