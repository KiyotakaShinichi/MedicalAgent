targetScope = 'resourceGroup'

@description('Short lowercase workload prefix.')
@minLength(3)
@maxLength(12)
param prefix string = 'nlcare'

@allowed([
  'dev'
  'staging'
])
param environment string = 'dev'

param location string = resourceGroup().location
param allowPublicNetworkAccess bool = false
param deployManagedSearch bool = false
param deployMessaging bool = false
param deployPostgres bool = false

@secure()
param postgresAdminPassword string = ''

param postgresAdminLogin string = 'nlcareadmin'

var suffix = uniqueString(resourceGroup().id)
var commonTags = {
  workload: 'nlcare'
  environment: environment
  clinicalValidation: 'false'
  healthcareProductionReady: 'false'
  patientDataAllowed: 'false'
}
var publicNetwork = allowPublicNetworkAccess ? 'Enabled' : 'Disabled'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-${environment}-logs-${suffix}'
  location: location
  tags: commonTags
  properties: {
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    publicNetworkAccessForIngestion: publicNetwork
    publicNetworkAccessForQuery: publicNetwork
  }
}

resource containerEnvironment 'Microsoft.App/managedEnvironments@2025-01-01' = {
  name: '${prefix}-${environment}-cae-${suffix}'
  location: location
  tags: commonTags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource dataLake 'Microsoft.Storage/storageAccounts@2025-01-01' = {
  name: take(replace('${prefix}${environment}lake${suffix}', '-', ''), 24)
  location: location
  tags: commonTags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowCrossTenantReplication: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    isHnsEnabled: true
    minimumTlsVersion: 'TLS1_2'
    publicNetworkAccess: publicNetwork
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2025-01-01' = {
  parent: dataLake
  name: 'default'
  properties: {
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource bronzeContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = {
  parent: blobService
  name: 'bronze'
  properties: {
    publicAccess: 'None'
  }
}

resource silverContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = {
  parent: blobService
  name: 'silver'
  properties: {
    publicAccess: 'None'
  }
}

resource goldContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = {
  parent: blobService
  name: 'gold'
  properties: {
    publicAccess: 'None'
  }
}

resource quarantineContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = {
  parent: blobService
  name: 'quarantine'
  properties: {
    publicAccess: 'None'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2025-05-01' = {
  name: take('${prefix}-${environment}-kv-${suffix}', 24)
  location: location
  tags: commonTags
  properties: {
    tenantId: tenant().tenantId
    enablePurgeProtection: true
    enableRbacAuthorization: true
    enableSoftDelete: true
    publicNetworkAccess: publicNetwork
    sku: {
      family: 'A'
      name: 'standard'
    }
    softDeleteRetentionInDays: 30
  }
}

resource managedSearch 'Microsoft.Search/searchServices@2025-05-01' = if (deployManagedSearch) {
  name: '${prefix}-${environment}-search-${suffix}'
  location: location
  tags: commonTags
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'basic'
  }
  properties: {
    disableLocalAuth: true
    hostingMode: 'default'
    partitionCount: 1
    publicNetworkAccess: publicNetwork
    replicaCount: 1
    semanticSearch: 'free'
  }
}

resource serviceBus 'Microsoft.ServiceBus/namespaces@2024-01-01' = if (deployMessaging) {
  name: '${prefix}-${environment}-sb-${suffix}'
  location: location
  tags: commonTags
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {
    disableLocalAuth: true
    publicNetworkAccess: publicNetwork
    zoneRedundant: false
  }
}

resource engineeringQueue 'Microsoft.ServiceBus/namespaces/queues@2024-01-01' = if (deployMessaging) {
  parent: serviceBus
  name: 'engineering-events'
  properties: {
    deadLetteringOnMessageExpiration: true
    defaultMessageTimeToLive: 'P1D'
    duplicateDetectionHistoryTimeWindow: 'P1D'
    enableBatchedOperations: true
    lockDuration: 'PT1M'
    maxDeliveryCount: 10
    requiresDuplicateDetection: true
  }
}

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = if (deployPostgres) {
  name: '${prefix}-${environment}-pg-${suffix}'
  location: location
  tags: commonTags
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: postgresAdminLogin
    administratorLoginPassword: postgresAdminPassword
    authConfig: {
      activeDirectoryAuth: 'Disabled'
      passwordAuth: 'Enabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: {
      publicNetworkAccess: publicNetwork
    }
    storage: {
      storageSizeGB: 32
    }
    version: '16'
  }
}

output architectureStatus string = 'reference-foundation-only'
output clinicalValidation bool = false
output healthcareProductionReady bool = false
output patientDataAllowed bool = false
output containerEnvironmentName string = containerEnvironment.name
output dataLakeName string = dataLake.name
output keyVaultName string = keyVault.name
output managedSearchName string = deployManagedSearch ? managedSearch.name : 'not-deployed'
output serviceBusName string = deployMessaging ? serviceBus.name : 'not-deployed'
output postgresName string = deployPostgres ? postgres.name : 'not-deployed'
