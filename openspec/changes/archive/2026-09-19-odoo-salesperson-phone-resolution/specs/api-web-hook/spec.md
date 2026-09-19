# Spec Delta: api-web-hook

## MODIFIED Requirements

### Requirement: Odoo Salesperson Resolution by Phone
The system SHALL query Odoo ERP on `res.partner` matching by valid phone fields (`phone` or `phone_mobile_search`) to resolve the salesperson `user_id`, resolving assigned salespersons for clients or internal salesperson users for employees, and returning `None` when no partner or salesperson is found.

#### Scenario: Existing Partner with Assigned Salesperson
- **WHEN** se recibe un número de teléfono que coincide en Odoo con un cliente asignado al vendedor con ID `5`
- **THEN** el resolvedor retorna `user_id=5`.

#### Scenario: Unregistered Phone Fallback
- **WHEN** se recibe un número de teléfono que no registra ninguna coincidencia en Odoo o cuyo cliente carece de comercial asignado
- **THEN** el resolvedor retorna `None` indicando la ausencia de autenticación sin asignar usuarios arbitrarios por defecto.

#### Scenario: Internal Salesperson Partner Resolution
- **WHEN** se recibe un número de teléfono perteneciente a un vendedor interno de Odoo (cuyo `res.partner` no tiene `user_id` asignado pero está vinculado a un usuario en `res.users`)
- **THEN** el resolvedor busca en `res.users` por `partner_id` y retorna el ID del propio usuario vendedor.
