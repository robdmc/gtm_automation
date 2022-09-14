#! /usr/bin/env bash

printenv | grep SALESFORCE | awk '{print "export", $1}'
printenv | grep '^PG' | awk '{print "export", $1}'
