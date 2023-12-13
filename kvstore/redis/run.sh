docker run -d --name fedlearn-kvstore -p 6379:6379 redis/redis-stack-server:7.2.0-v6
docker logs --follow fedlearn-kvstore
