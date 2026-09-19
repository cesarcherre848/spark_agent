# Spec Delta: api-web-hook

## MODIFIED Requirements

### Requirement: Odoo Salesperson Resolution by Phone
The system SHALL query Odoo ERP exclusively for internal salesperson users (`res.users` with `active=True` and `share=False`) using exact phone number matching (`=`), returning their `user_id` when found and `None` for non-salespeople or unregistered numbers.

#### Scenario: Existing Partner with Assigned Salesperson
- **WHEN** se recibe un número de teléfono perteneciente a un contacto que no es un vendedor interno de Odoo (incluso si es un cliente)
- **THEN** el resolvedor retorna `None` denegando el acceso ya que el agente es exclusivo para vendedores.

#### Scenario: Unregistered Phone Fallback
- **WHEN** se recibe un número de teléfono que no registra ninguna coincidencia exacta con un vendedor en Odoo
- **THEN** el resolvedor retorna `None` indicando la ausencia de autenticación sin asignar usuarios arbitrarios por defecto.

#### Scenario: Internal Salesperson Partner Resolution
- **WHEN** se recibe un número de teléfono que coincide exactamente con el teléfono de un vendedor interno activo de Odoo (`res.users` con `share=False`)
- **THEN** el resolvedor retorna el ID de dicho vendedor en `res.users`.
