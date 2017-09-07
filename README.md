# Python client for TORCS with network plugin for the 2012 SCRC

This is a reimplementation of the original SCRC client pySrcrcClient from lanquarden. It is used for teaching Python.

## `Client`

* top level class
* handles _all_ aspects of networking (connection management, encoding)
* decodes class `State` from message from server, `state = self.decode(msg)`
* encodes class `Command` for message to server, `msg = self.encode(command)`
* internal state connection properties only and driver instance


## `Driver`

* encapsulates driving logic only
* main entry point: `drive(state: State) -> Command`

## `State`

* represents the incoming car state

## `Command`

* holds the outgoing driving command
