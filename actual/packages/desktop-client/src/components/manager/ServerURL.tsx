import React from 'react';
import { Trans } from 'react-i18next';
import { useLocation } from 'react-router';

import { Text } from '@actual-app/components/text';
import { View } from '@actual-app/components/view';

import { Link } from '#components/common/Link';
import { useServerURL } from '#components/ServerContext';

export function ServerURL() {
  const url = useServerURL();
  const location = useLocation();

  if (
    location.pathname.startsWith('/login') ||
    location.pathname.startsWith('/bootstrap')
  ) {
    return null;
  }

  return (
    <View
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        justifyContent: 'center',
        flexDirection: 'row',
        marginBottom: 15,
        zIndex: 5000,
      }}
    >
      <Text>
        {url ? (
          <Trans>
            Using server: <strong>{url}</strong>
          </Trans>
        ) : (
          <Trans>
            <strong>No server configured</strong>
          </Trans>
        )}
      </Text>
      <Link variant="internal" to="/config-server" style={{ marginLeft: 15 }}>
        <Trans>Change</Trans>
      </Link>
    </View>
  );
}
