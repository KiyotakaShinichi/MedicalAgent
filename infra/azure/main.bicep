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

@description('Public endpoints remain disabled unless a disposable development deployment explicitly opts in.')
param allowPublicNetworkAccess bool = false

@description('Deploy the Container Apps environment. Application revisions are intentionally out of scope.')
param deployComputeEnvironment bool = false

@description('Deploy an internal backend Container App revision. Disabled until a reviewed immutable image is supplied.')
param deployApplication bool = false

@description('Immutable backend container image reference, preferably pinned by digest.')
param backendContainerImage string = ''

@description('Optional Key Vault secret URI for DATABASE_URL. The workload identity must be granted secret access.')
param databaseUrlSecretUri string = ''

@description('Deploy a VNet, private DNS zones, and private endpoints for enabled services.')
param deployPrivateNetworking bool = false

param deployManagedSearch bool = false
param deployMessaging bool = false
param deployPostgres bool = false

@description('Create an engineering-only action group and resource-group deployment-failure alert.')
param deployOperationalAlerts bool = false

@description('Create a resource-group-scoped monthly budget. A contact email is required when enabled.')
param deployCostControls bool = false

@minValue(1)
param monthlyBudgetAmount int = 50

@description('Engineering alert recipient. Never use a patient address.')
param operationsContactEmail string = ''

@description('Budget start date must be the first day of a month.')
param budgetStartDate string = utcNow('yyyy-MM-01')

@secure()
param postgresAdminPassword string = ''

@minLength(3)
param postgresAdminLogin string = 'nlcareadmin'

@minValue(7)
@maxValue(35)
param postgresBackupRetentionDays int = 14

@description('Geo-redundant PostgreSQL backup is opt-in because availability and cost vary by region.')
param postgresGeoRedundantBackup bool = false

var suffix = uniqueString(resourceGroup().id)
var commonTags = {
  workload: 'nlcare'
  environment: environment
  clinicalValidation: 'false'
  healthcareProductionReady: 'false'
  patientDataAllowed: 'false'
  dataScope: 'curated-non-patient-only'
}
var publicNetwork = allowPublicNetworkAccess ? 'Enabled' : 'Disabled'
var deployContainerEnvironment = deployComputeEnvironment || deployApplication
var storageBlobDataContributorRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
)
var keyVaultSecretsUserRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4633458b-17de-408a-b874-0445c86b69e6'
)
var searchIndexDataContributorRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
)
var serviceBusDataSenderRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '69a216fc-b8fb-44d8-bc22-1f3c2cd27a39'
)
var serviceBusDataReceiverRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4f6c032b-1dd4-4a5e-8adf-73048f2f1f47'
)

resource workloadIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2024-11-30' = {
  name: '${prefix}-${environment}-identity-${suffix}'
  location: location
  tags: commonTags
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-05-01' = if (deployPrivateNetworking) {
  name: '${prefix}-${environment}-vnet-${suffix}'
  location: location
  tags: commonTags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.42.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'container-apps'
        properties: {
          addressPrefix: '10.42.0.0/23'
          delegations: [
            {
              name: 'container-apps-delegation'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
      {
        name: 'private-endpoints'
        properties: {
          addressPrefix: '10.42.4.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'postgres'
        properties: {
          addressPrefix: '10.42.5.0/24'
          delegations: [
            {
              name: 'postgres-delegation'
              properties: {
                serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
              }
            }
          ]
        }
      }
    ]
  }
}

resource containerAppsSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' existing = if (deployPrivateNetworking) {
  name: 'container-apps'
  parent: virtualNetwork
}

resource privateEndpointsSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' existing = if (deployPrivateNetworking) {
  name: 'private-endpoints'
  parent: virtualNetwork
}

resource postgresSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' existing = if (deployPrivateNetworking) {
  name: 'postgres'
  parent: virtualNetwork
}

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

resource containerEnvironment 'Microsoft.App/managedEnvironments@2025-01-01' = if (deployContainerEnvironment) {
  name: '${prefix}-${environment}-cae-${suffix}'
  location: location
  tags: commonTags
  properties: union(
    {
      appLogsConfiguration: {
        destination: 'log-analytics'
        logAnalyticsConfiguration: {
          customerId: logAnalytics.properties.customerId
          sharedKey: logAnalytics.listKeys().primarySharedKey
        }
      }
    },
    deployPrivateNetworking
      ? {
          vnetConfiguration: {
            infrastructureSubnetId: containerAppsSubnet.id
            internal: true
          }
        }
      : {}
  )
}

resource backendContainerApp 'Microsoft.App/containerApps@2025-01-01' = if (deployApplication) {
  name: '${prefix}-${environment}-api-${suffix}'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${workloadIdentity.id}': {}
    }
  }
  properties: {
    environmentId: containerEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        allowInsecure: false
        external: false
        targetPort: 8000
        transport: 'auto'
      }
      secrets: empty(databaseUrlSecretUri)
        ? []
        : [
            {
              identity: workloadIdentity.id
              keyVaultUrl: databaseUrlSecretUri
              name: 'database-url'
            }
          ]
    }
    template: {
      containers: [
        {
          env: empty(databaseUrlSecretUri)
            ? [
                {
                  name: 'ENVIRONMENT'
                  value: environment
                }
              ]
            : [
                {
                  name: 'ENVIRONMENT'
                  value: environment
                }
                {
                  name: 'DATABASE_URL'
                  secretRef: 'database-url'
                }
              ]
          image: backendContainerImage
          name: 'api'
          probes: [
            {
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 15
              timeoutSeconds: 5
              type: 'Liveness'
            }
            {
              httpGet: {
                path: '/ready'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 15
              timeoutSeconds: 5
              type: 'Readiness'
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
      scale: {
        maxReplicas: 2
        minReplicas: 0
        rules: [
          {
            custom: {
              metadata: {
                concurrentRequests: '20'
              }
              type: 'http'
            }
            name: 'http-concurrency'
          }
        ]
      }
    }
  }
}

resource dataLake 'Microsoft.Storage/storageAccounts@2025-01-01' = {
  #disable-next-line BCP334
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
      days: 14
    }
    deleteRetentionPolicy: {
      enabled: true
      days: 14
    }
    isVersioningEnabled: true
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
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
    disableLocalAuth: true
    hostingMode: 'Default'
    networkRuleSet: {
      bypass: 'None'
      ipRules: []
    }
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

resource postgresPrivateDns 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployPrivateNetworking && deployPostgres) {
  name: '${prefix}-${environment}.postgres.database.azure.com'
  location: 'global'
  tags: commonTags
}

resource postgresPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployPrivateNetworking && deployPostgres) {
  parent: postgresPrivateDns
  name: 'postgres-vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
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
      backupRetentionDays: postgresBackupRetentionDays
      geoRedundantBackup: postgresGeoRedundantBackup ? 'Enabled' : 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: deployPrivateNetworking
      ? {
          delegatedSubnetResourceId: postgresSubnet.id
          privateDnsZoneArmResourceId: postgresPrivateDns.id
        }
      : {
          publicNetworkAccess: publicNetwork
        }
    storage: {
      autoGrow: 'Enabled'
      storageSizeGB: 32
    }
    version: '16'
  }
  dependsOn: [
    postgresPrivateDnsLink
  ]
}

resource blobPrivateDns 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployPrivateNetworking) {
  name: 'privatelink.blob.${az.environment().suffixes.storage}'
  location: 'global'
  tags: commonTags
}

resource dfsPrivateDns 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployPrivateNetworking) {
  name: 'privatelink.dfs.${az.environment().suffixes.storage}'
  location: 'global'
  tags: commonTags
}

resource vaultPrivateDns 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployPrivateNetworking) {
  name: 'privatelink.vaultcore.azure.net'
  location: 'global'
  tags: commonTags
}

resource searchPrivateDns 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployPrivateNetworking && deployManagedSearch) {
  name: 'privatelink.search.windows.net'
  location: 'global'
  tags: commonTags
}

resource serviceBusPrivateDns 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployPrivateNetworking && deployMessaging) {
  name: 'privatelink.servicebus.windows.net'
  location: 'global'
  tags: commonTags
}

resource blobPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployPrivateNetworking) {
  parent: blobPrivateDns
  name: 'blob-vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource dfsPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployPrivateNetworking) {
  parent: dfsPrivateDns
  name: 'dfs-vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource vaultPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployPrivateNetworking) {
  parent: vaultPrivateDns
  name: 'vault-vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource searchPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployPrivateNetworking && deployManagedSearch) {
  parent: searchPrivateDns
  name: 'search-vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource serviceBusPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployPrivateNetworking && deployMessaging) {
  parent: serviceBusPrivateDns
  name: 'servicebus-vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource blobPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployPrivateNetworking) {
  name: '${prefix}-${environment}-blob-pe-${suffix}'
  location: location
  tags: commonTags
  properties: {
    subnet: {
      id: privateEndpointsSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'blob'
        properties: {
          groupIds: [
            'blob'
          ]
          privateLinkServiceId: dataLake.id
        }
      }
    ]
  }
}

resource blobPrivateDnsGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (deployPrivateNetworking) {
  parent: blobPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'blob'
        properties: {
          privateDnsZoneId: blobPrivateDns.id
        }
      }
    ]
  }
}

resource dfsPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployPrivateNetworking) {
  name: '${prefix}-${environment}-dfs-pe-${suffix}'
  location: location
  tags: commonTags
  properties: {
    subnet: {
      id: privateEndpointsSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'dfs'
        properties: {
          groupIds: [
            'dfs'
          ]
          privateLinkServiceId: dataLake.id
        }
      }
    ]
  }
}

resource dfsPrivateDnsGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (deployPrivateNetworking) {
  parent: dfsPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'dfs'
        properties: {
          privateDnsZoneId: dfsPrivateDns.id
        }
      }
    ]
  }
}

resource vaultPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployPrivateNetworking) {
  name: '${prefix}-${environment}-vault-pe-${suffix}'
  location: location
  tags: commonTags
  properties: {
    subnet: {
      id: privateEndpointsSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'vault'
        properties: {
          groupIds: [
            'vault'
          ]
          privateLinkServiceId: keyVault.id
        }
      }
    ]
  }
}

resource vaultPrivateDnsGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (deployPrivateNetworking) {
  parent: vaultPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'vault'
        properties: {
          privateDnsZoneId: vaultPrivateDns.id
        }
      }
    ]
  }
}

resource searchPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployPrivateNetworking && deployManagedSearch) {
  name: '${prefix}-${environment}-search-pe-${suffix}'
  location: location
  tags: commonTags
  properties: {
    subnet: {
      id: privateEndpointsSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'search'
        properties: {
          groupIds: [
            'searchService'
          ]
          privateLinkServiceId: managedSearch.id
        }
      }
    ]
  }
}

resource searchPrivateDnsGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (deployPrivateNetworking && deployManagedSearch) {
  parent: searchPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'search'
        properties: {
          privateDnsZoneId: searchPrivateDns.id
        }
      }
    ]
  }
}

resource serviceBusPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployPrivateNetworking && deployMessaging) {
  name: '${prefix}-${environment}-servicebus-pe-${suffix}'
  location: location
  tags: commonTags
  properties: {
    subnet: {
      id: privateEndpointsSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'namespace'
        properties: {
          groupIds: [
            'namespace'
          ]
          privateLinkServiceId: serviceBus.id
        }
      }
    ]
  }
}

resource serviceBusPrivateDnsGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (deployPrivateNetworking && deployMessaging) {
  parent: serviceBusPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'servicebus'
        properties: {
          privateDnsZoneId: serviceBusPrivateDns.id
        }
      }
    ]
  }
}

resource storageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(dataLake.id, workloadIdentity.id, storageBlobDataContributorRoleId)
  scope: dataLake
  properties: {
    principalId: workloadIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataContributorRoleId
  }
}

resource vaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, workloadIdentity.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    principalId: workloadIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRoleId
  }
}

resource searchRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployManagedSearch) {
  name: guid(managedSearch.id, workloadIdentity.id, searchIndexDataContributorRoleId)
  scope: managedSearch
  properties: {
    principalId: workloadIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: searchIndexDataContributorRoleId
  }
}

resource serviceBusSenderRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployMessaging) {
  name: guid(serviceBus.id, workloadIdentity.id, serviceBusDataSenderRoleId)
  scope: serviceBus
  properties: {
    principalId: workloadIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: serviceBusDataSenderRoleId
  }
}

resource serviceBusReceiverRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployMessaging) {
  name: guid(serviceBus.id, workloadIdentity.id, serviceBusDataReceiverRoleId)
  scope: serviceBus
  properties: {
    principalId: workloadIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: serviceBusDataReceiverRoleId
  }
}

resource operationsActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = if (deployOperationalAlerts && !empty(operationsContactEmail)) {
  name: '${prefix}-${environment}-engineering-alerts'
  location: 'global'
  tags: commonTags
  properties: {
    enabled: true
    groupShortName: 'nlcareeng'
    emailReceivers: [
      {
        emailAddress: operationsContactEmail
        name: 'engineering-operator'
        useCommonAlertSchema: true
      }
    ]
  }
}

resource deploymentFailureAlert 'Microsoft.Insights/activityLogAlerts@2020-10-01' = if (deployOperationalAlerts && !empty(operationsContactEmail)) {
  name: '${prefix}-${environment}-deployment-failures'
  location: 'global'
  tags: commonTags
  properties: {
    actions: {
      actionGroups: [
        {
          actionGroupId: operationsActionGroup.id
        }
      ]
    }
    condition: {
      allOf: [
        {
          field: 'category'
          equals: 'Administrative'
        }
        {
          field: 'status'
          equals: 'Failed'
        }
      ]
    }
    description: 'Engineering-only alert for failed Azure control-plane operations.'
    enabled: true
    scopes: [
      resourceGroup().id
    ]
  }
}

resource engineeringBudget 'Microsoft.Consumption/budgets@2024-08-01' = if (deployCostControls && !empty(operationsContactEmail)) {
  name: '${prefix}-${environment}-monthly-budget'
  properties: {
    amount: monthlyBudgetAmount
    category: 'Cost'
    notifications: {
      Actual80Percent: {
        contactEmails: [
          operationsContactEmail
        ]
        contactGroups: []
        contactRoles: []
        enabled: true
        locale: 'en-us'
        operator: 'GreaterThanOrEqualTo'
        threshold: 80
        thresholdType: 'Actual'
      }
      Forecast100Percent: {
        contactEmails: [
          operationsContactEmail
        ]
        contactGroups: []
        contactRoles: []
        enabled: true
        locale: 'en-us'
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Forecasted'
      }
    }
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: '${budgetStartDate}T00:00:00Z'
    }
  }
}

output architectureStatus string = 'reference-foundation-only'
output clinicalValidation bool = false
output healthcareProductionReady bool = false
output patientDataAllowed bool = false
output workloadIdentityName string = workloadIdentity.name
output containerEnvironmentName string = deployContainerEnvironment ? containerEnvironment.name : 'not-deployed'
output backendContainerAppName string = deployApplication ? backendContainerApp.name : 'not-deployed'
output dataLakeName string = dataLake.name
output keyVaultName string = keyVault.name
output managedSearchName string = deployManagedSearch ? managedSearch.name : 'not-deployed'
output serviceBusName string = deployMessaging ? serviceBus.name : 'not-deployed'
output postgresName string = deployPostgres ? postgres.name : 'not-deployed'
output privateNetworkingEnabled bool = deployPrivateNetworking
output costControlsEnabled bool = deployCostControls && !empty(operationsContactEmail)
output operationalAlertsEnabled bool = deployOperationalAlerts && !empty(operationsContactEmail)
