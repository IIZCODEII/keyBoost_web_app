mkdir -p ~/.keyboost_web/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.keyboost_web/config.toml
