#!/bin/bash

# Main project directory
mkdir -p conversational-analytics
cd conversational-analytics

# Backend Structure
echo "Creating backend structure..."
mkdir -p backend/{config,core/{schema_detection,query_generation,visualization},services,models,api/{routes,middleware},utils}
touch backend/app.py
touch backend/requirements.txt
touch backend/config/{__init__.py,settings.py,database.py}
touch backend/core/{__init__.py}
touch backend/core/schema_detection/{__init__.py,detector.py,field_analyzer.py,relationship_finder.py,statistics_engine.py,collection_classifier.py}
touch backend/core/query_generation/{__init__.py,gemini_client.py,prompt_builder.py,query_validator.py,fallback_generator.py}
touch backend/core/visualization/{__init__.py,chart_analyzer.py,chart_suggester.py,chart_generator.py,response_formatter.py}
touch backend/services/{__init__.py,analytics_service.py,schema_service.py,cache_service.py,database_service.py,gemini_service.py}
touch backend/models/{__init__.py,schema_models.py,query_models.py,chart_models.py,response_models.py}
touch backend/api/{__init__.py}
touch backend/api/routes/{__init__.py,analytics.py,schema.py,charts.py,health.py}
touch backend/api/middleware/{__init__.py,cors.py,rate_limiting.py,error_handling.py}
touch backend/utils/{__init__.py,logging_config.py,validators.py,helpers.py,constants.py}

# Frontend Structure
echo "Creating frontend structure..."
mkdir -p frontend/{src/{components/{common,chat,charts,responses},services,hooks,utils,styles},public}
touch frontend/src/App.jsx
touch frontend/src/main.jsx
touch frontend/src/components/common/{Header.jsx,Footer.jsx,Loading.jsx,ErrorBoundary.jsx}
touch frontend/src/components/chat/{ChatInterface.jsx,MessageBubble.jsx,InputBox.jsx,SuggestionChips.jsx}
touch frontend/src/components/charts/{ChartContainer.jsx,ChartButton.jsx,BarChart.jsx,LineChart.jsx,PieChart.jsx,ChartTypeSelector.jsx}
touch frontend/src/components/responses/{TextResponse.jsx,DataTable.jsx,MetricCard.jsx,ResponseActions.jsx}
touch frontend/src/services/{api.js,chartService.js,responseParser.js}
touch frontend/src/hooks/{useChat.js,useCharts.js,useApi.js}
touch frontend/src/utils/{formatters.js,validators.js,constants.js}
touch frontend/src/styles/{globals.css,components.css,charts.css}
touch frontend/public/{index.html,favicon.ico}
touch frontend/{package.json,vite.config.js}

# Docs Structure
echo "Creating docs structure..."
mkdir -p docs
touch docs/{API.md,SCHEMA_DETECTION.md,DEPLOYMENT.md,EXAMPLES.md}

# Tests Structure
echo "Creating tests structure..."
mkdir -p tests/{backend,frontend/{components,services}}
touch tests/backend/{test_schema_detection.py,test_query_generation.py,test_chart_analysis.py,test_api_endpoints.py}

# Docker Structure
echo "Creating docker structure..."
mkdir -p docker
touch docker/{Dockerfile.backend,Dockerfile.frontend,docker-compose.yml}

# Root files
echo "Creating root files..."
touch .env.example .gitignore README.md pyproject.toml

echo "Project structure for conversational-analytics created successfully!"