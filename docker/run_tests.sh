#!/usr/bin/env bash
find . -name "test_*" -type d -exec sh -c '(cd {} && docker-compose run test)' ';'
