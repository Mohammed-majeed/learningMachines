# We base ourselves on Unbuntu. This is the base OS we are installing.
FROM ubuntu:20.04

# Run apt-get to install git
# This has three stages:
#  * First, we update apt, to make sure we are installing the latest version of everything
#  * Second, we are installing git
#  * Third, we are removing some junk files that were created in the process.
#    (You don't have to bother about this third step)
RUN apt-get -y update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

CMD ["echo","Hello, World"]