export type NewUserEntity = {
  userName: string;
  displayName: string;
  role: string;
  enabled: boolean;
  password?: string;
};

export type UserEntity = {
  id: string;
  owner: boolean;
} & NewUserEntity;

export type UserEntityDropdown = {
  userId: string;
  userName: string;
  displayName?: string;
};

export type UserAvailable = {
  userId: string;
  displayName?: string;
  userName: string;
  haveAccess?: number;
  owner?: number;
};
