#! /usr/bin/env bash

printenv | grep SALESFORCE | awk '{print "export", $1}'
