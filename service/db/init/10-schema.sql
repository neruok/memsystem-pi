\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CREATE ON SCHEMA public TO memsystem_owner;
GRANT USAGE ON SCHEMA public TO memsystem_service;
SET ROLE memsystem_owner;

CREATE TYPE membership_state AS ENUM ('active', 'suspended', 'revoked');
CREATE TYPE membership_role AS ENUM ('member', 'admin');
CREATE TYPE agent_run_state AS ENUM ('active', 'expired', 'revoked');
CREATE TYPE compartment_scope AS ENUM ('tenant', 'global', 'user', 'agent', 'workspace', 'project');
CREATE TYPE document_kind AS ENUM ('collection', 'page', 'journal');
CREATE TYPE author_kind AS ENUM ('user', 'agent', 'system');
CREATE TYPE link_kind AS ENUM ('references', 'supports', 'contradicts', 'supersedes', 'related');
CREATE TYPE embedding_state AS ENUM ('pending', 'ready', 'failed');
CREATE TYPE job_kind AS ENUM ('embed', 'index_add', 'index_remove');
CREATE TYPE job_state AS ENUM ('pending', 'running', 'complete', 'failed');

CREATE TABLE tenants (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tenant_memberships (
    tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    role membership_role NOT NULL DEFAULT 'member',
    state membership_state NOT NULL DEFAULT 'active',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE agents (
    tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    key text NOT NULL CHECK (char_length(key) BETWEEN 1 AND 200),
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    archived_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, key)
);

CREATE TABLE agent_delegations (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    agent_id uuid NOT NULL,
    can_read boolean NOT NULL DEFAULT true,
    can_write boolean NOT NULL DEFAULT false,
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, user_id)
        REFERENCES tenant_memberships (tenant_id, user_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, agent_id)
        REFERENCES agents (tenant_id, id) ON DELETE CASCADE,
    CHECK (can_read OR NOT can_write)
);

CREATE TABLE agent_runs (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    agent_id uuid NOT NULL,
    session_key text NOT NULL CHECK (char_length(session_key) BETWEEN 1 AND 500),
    state agent_run_state NOT NULL DEFAULT 'active',
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, session_key),
    FOREIGN KEY (tenant_id, agent_id)
        REFERENCES agents (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE workspaces (
    tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    key text NOT NULL CHECK (char_length(key) BETWEEN 1 AND 200),
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0),
    archived_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, key)
);

CREATE TABLE workspace_memberships (
    tenant_id uuid NOT NULL,
    workspace_id uuid NOT NULL,
    user_id uuid NOT NULL,
    can_read boolean NOT NULL DEFAULT true,
    can_write boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id, user_id),
    FOREIGN KEY (tenant_id, workspace_id)
        REFERENCES workspaces (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, user_id)
        REFERENCES tenant_memberships (tenant_id, user_id) ON DELETE CASCADE,
    CHECK (can_read OR NOT can_write)
);

CREATE TABLE projects (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL,
    key text NOT NULL CHECK (char_length(key) BETWEEN 1 AND 200),
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    root_hint text CHECK (root_hint IS NULL OR char_length(root_hint) <= 1000),
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0),
    archived_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, workspace_id, key),
    UNIQUE (tenant_id, workspace_id, id),
    FOREIGN KEY (tenant_id, workspace_id)
        REFERENCES workspaces (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE project_memberships (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    user_id uuid NOT NULL,
    can_read boolean NOT NULL DEFAULT true,
    can_write boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, user_id),
    FOREIGN KEY (tenant_id, project_id)
        REFERENCES projects (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, user_id)
        REFERENCES tenant_memberships (tenant_id, user_id) ON DELETE CASCADE,
    CHECK (can_read OR NOT can_write)
);

CREATE TABLE mutation_idempotency (
    tenant_id uuid NOT NULL,
    user_id uuid NOT NULL,
    key text NOT NULL CHECK (char_length(key) BETWEEN 1 AND 200),
    operation text NOT NULL CHECK (char_length(operation) BETWEEN 1 AND 100),
    request_hash bytea NOT NULL CHECK (octet_length(request_hash) = 32),
    response jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, key),
    FOREIGN KEY (tenant_id, user_id)
        REFERENCES tenant_memberships (tenant_id, user_id) ON DELETE CASCADE,
    CHECK (response IS NULL OR
           (jsonb_typeof(response) = 'object' AND octet_length(response::text) <= 65536))
);

CREATE TABLE compartments (
    tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    parent_id uuid,
    path ltree NOT NULL,
    scope_type compartment_scope NOT NULL,
    scope_id uuid,
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, path),
    UNIQUE (tenant_id, scope_type, scope_id),
    FOREIGN KEY (tenant_id, parent_id)
        REFERENCES compartments (tenant_id, id),
    CHECK (parent_id IS NULL OR parent_id <> id),
    CHECK ((scope_type IN ('tenant', 'global')) = (scope_id IS NULL)),
    CHECK (scope_type <> 'tenant' OR parent_id IS NULL),
    CHECK (scope_type = 'tenant' OR parent_id IS NOT NULL)
);

CREATE UNIQUE INDEX one_tenant_root
    ON compartments (tenant_id) WHERE scope_type = 'tenant';
CREATE UNIQUE INDEX one_global_compartment
    ON compartments (tenant_id) WHERE scope_type = 'global';
CREATE INDEX compartments_path_gist ON compartments USING gist (path);

CREATE TABLE subsystems (
    tenant_id uuid NOT NULL,
    workspace_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    key text NOT NULL CHECK (char_length(key) BETWEEN 1 AND 200),
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    description text NOT NULL CHECK (char_length(description) BETWEEN 1 AND 4000),
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0),
    archived_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, workspace_id, key),
    UNIQUE (tenant_id, workspace_id, id),
    FOREIGN KEY (tenant_id, workspace_id)
        REFERENCES workspaces (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE project_subsystems (
    tenant_id uuid NOT NULL,
    workspace_id uuid NOT NULL,
    project_id uuid NOT NULL,
    subsystem_id uuid NOT NULL,
    PRIMARY KEY (tenant_id, project_id, subsystem_id),
    FOREIGN KEY (tenant_id, workspace_id, project_id)
        REFERENCES projects (tenant_id, workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, workspace_id, subsystem_id)
        REFERENCES subsystems (tenant_id, workspace_id, id) ON DELETE CASCADE
);

CREATE TABLE documents (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    compartment_id uuid NOT NULL,
    parent_id uuid,
    kind document_kind NOT NULL,
    slug text NOT NULL CHECK (char_length(slug) BETWEEN 1 AND 200),
    current_revision_id uuid,
    expires_at timestamptz,
    deleted_at timestamptz,
    auto_recall_approved_at timestamptz,
    auto_recall_approved_by uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, compartment_id, id),
    FOREIGN KEY (tenant_id, compartment_id)
        REFERENCES compartments (tenant_id, id),
    FOREIGN KEY (tenant_id, compartment_id, parent_id)
        REFERENCES documents (tenant_id, compartment_id, id),
    CHECK (parent_id IS NULL OR parent_id <> id)
);

CREATE UNIQUE INDEX live_root_document_slugs
    ON documents (tenant_id, compartment_id, slug)
    WHERE parent_id IS NULL AND deleted_at IS NULL;
CREATE UNIQUE INDEX live_child_document_slugs
    ON documents (tenant_id, compartment_id, parent_id, slug)
    WHERE parent_id IS NOT NULL AND deleted_at IS NULL;

CREATE TABLE document_revisions (
    tenant_id uuid NOT NULL,
    document_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    revision integer NOT NULL CHECK (revision > 0),
    title text NOT NULL CHECK (char_length(title) BETWEEN 1 AND 500),
    markdown text NOT NULL CHECK (octet_length(markdown) <= 1048576),
    created_by_type author_kind NOT NULL,
    created_by_id uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, document_id, id),
    UNIQUE (tenant_id, document_id, revision),
    FOREIGN KEY (tenant_id, document_id)
        REFERENCES documents (tenant_id, id) ON DELETE CASCADE,
    CHECK ((created_by_type = 'system') = (created_by_id IS NULL))
);

ALTER TABLE documents ADD CONSTRAINT documents_current_revision_fk
    FOREIGN KEY (tenant_id, id, current_revision_id)
    REFERENCES document_revisions (tenant_id, document_id, id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE document_subsystems (
    tenant_id uuid NOT NULL,
    document_id uuid NOT NULL,
    subsystem_id uuid NOT NULL,
    PRIMARY KEY (tenant_id, document_id, subsystem_id),
    FOREIGN KEY (tenant_id, document_id)
        REFERENCES documents (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, subsystem_id)
        REFERENCES subsystems (tenant_id, id) ON DELETE CASCADE
);

CREATE SEQUENCE chunk_vector_id_seq AS bigint MINVALUE 1;

CREATE TABLE chunks (
    tenant_id uuid NOT NULL,
    document_id uuid NOT NULL,
    revision_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    vector_id bigint NOT NULL DEFAULT nextval('chunk_vector_id_seq'),
    heading_path text[] NOT NULL DEFAULT '{}',
    text text NOT NULL CHECK (octet_length(text) <= 262144),
    position integer NOT NULL CHECK (position >= 0),
    source_start integer NOT NULL CHECK (source_start >= 0),
    source_end integer NOT NULL CHECK (source_end >= source_start),
    chunk_profile text NOT NULL CHECK (char_length(chunk_profile) BETWEEN 1 AND 100),
    search_document tsvector NOT NULL,
    embedding vector,
    embedding_model text,
    embedding_version text,
    embedding_dimension integer,
    embedding_state embedding_state NOT NULL DEFAULT 'pending',
    embedding_error text,
    embedding_attempted_at timestamptz,
    indexed_revision integer,
    indexed_generation bigint,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (vector_id),
    UNIQUE (tenant_id, revision_id, position),
    FOREIGN KEY (tenant_id, document_id, revision_id)
        REFERENCES document_revisions (tenant_id, document_id, id) ON DELETE CASCADE,
    CHECK (vector_id > 0),
    CHECK (embedding_dimension IS NULL OR
           (embedding_dimension > 0 AND embedding_dimension <= 16384 AND embedding_dimension % 8 = 0)),
    CHECK ((embedding IS NULL) = (embedding_dimension IS NULL)),
    CHECK (embedding IS NULL OR vector_dims(embedding) = embedding_dimension),
    CHECK (embedding_state <> 'ready' OR embedding IS NOT NULL)
);

CREATE INDEX chunks_search_gin ON chunks USING gin (search_document);

CREATE TABLE document_links (
    tenant_id uuid NOT NULL,
    source_document_id uuid NOT NULL,
    target_document_id uuid NOT NULL,
    type link_kind NOT NULL,
    weight double precision NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, source_document_id, target_document_id, type),
    FOREIGN KEY (tenant_id, source_document_id)
        REFERENCES documents (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, target_document_id)
        REFERENCES documents (tenant_id, id) ON DELETE CASCADE,
    CHECK (source_document_id <> target_document_id),
    CHECK (weight BETWEEN 0 AND 1 AND weight <> 'NaN'::double precision)
);

CREATE SEQUENCE job_enqueue_seq AS bigint MINVALUE 1;

CREATE TABLE jobs (
    tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    enqueue_seq bigint NOT NULL DEFAULT nextval('job_enqueue_seq'),
    document_id uuid,
    revision_id uuid,
    chunk_id uuid,
    vector_id bigint,
    profile text NOT NULL CHECK (char_length(profile) BETWEEN 1 AND 100),
    kind job_kind NOT NULL,
    state job_state NOT NULL DEFAULT 'pending',
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    error text,
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    claimed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    CHECK (vector_id IS NULL OR vector_id > 0),
    CHECK (
        (kind IN ('embed', 'index_add') AND document_id IS NOT NULL AND
         revision_id IS NOT NULL AND chunk_id IS NOT NULL AND vector_id IS NOT NULL) OR
        (kind = 'index_remove' AND vector_id IS NOT NULL)
    )
);

CREATE INDEX claimable_jobs
    ON jobs (next_attempt_at, created_at)
    WHERE state IN ('pending', 'failed');

CREATE INDEX jobs_replay_order
    ON jobs (tenant_id, profile, enqueue_seq);

CREATE FUNCTION validate_job_target() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.kind IN ('embed', 'index_add') AND NOT EXISTS (
        SELECT 1 FROM public.chunks
        WHERE tenant_id = NEW.tenant_id
          AND id = NEW.chunk_id
          AND document_id = NEW.document_id
          AND revision_id = NEW.revision_id
          AND vector_id = NEW.vector_id
    ) THEN
        RAISE EXCEPTION 'job target does not identify one chunk';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER jobs_validate_target
BEFORE INSERT OR UPDATE OF tenant_id, kind, document_id, revision_id, chunk_id, vector_id ON jobs
FOR EACH ROW EXECUTE FUNCTION validate_job_target();

CREATE TABLE vector_index_state (
    tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    profile text NOT NULL CHECK (char_length(profile) BETWEEN 1 AND 100),
    generation bigint NOT NULL CHECK (generation > 0),
    generation_path text NOT NULL CHECK (char_length(generation_path) BETWEEN 1 AND 2000),
    checksum text NOT NULL CHECK (char_length(checksum) BETWEEN 1 AND 200),
    job_high_water_mark bigint,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, profile)
);

CREATE FUNCTION current_tenant_id() RETURNS uuid
LANGUAGE sql STABLE PARALLEL SAFE
RETURN nullif(current_setting('memsystem.tenant_id', true), '')::uuid;

CREATE FUNCTION validate_compartment_hierarchy() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    parent_scope compartment_scope;
    parent_scope_id uuid;
    parent_path ltree;
    project_workspace_id uuid;
BEGIN
    IF NEW.scope_type = 'tenant' THEN
        IF nlevel(NEW.path) <> 1 THEN
            RAISE EXCEPTION 'tenant compartment path must have one label';
        END IF;
        RETURN NEW;
    END IF;

    SELECT scope_type, scope_id, path INTO parent_scope, parent_scope_id, parent_path
    FROM public.compartments
    WHERE tenant_id = NEW.tenant_id AND id = NEW.parent_id
    FOR KEY SHARE;

    IF NOT FOUND OR nlevel(NEW.path) <> nlevel(parent_path) + 1 THEN
        RAISE EXCEPTION 'compartment path must be directly below its parent';
    END IF;
    IF subpath(NEW.path, 0, nlevel(NEW.path) - 1) <> parent_path THEN
        RAISE EXCEPTION 'compartment path must be directly below its parent';
    END IF;

    IF (NEW.scope_type = 'project' AND parent_scope <> 'workspace') OR
       (NEW.scope_type <> 'project' AND parent_scope <> 'tenant') THEN
        RAISE EXCEPTION 'invalid compartment parent type';
    END IF;

    IF NEW.scope_type = 'user' AND NOT EXISTS (
        SELECT 1 FROM public.tenant_memberships
        WHERE tenant_id = NEW.tenant_id AND user_id = NEW.scope_id
    ) THEN
        RAISE EXCEPTION 'user compartment target does not exist';
    ELSIF NEW.scope_type = 'agent' AND NOT EXISTS (
        SELECT 1 FROM public.agents
        WHERE tenant_id = NEW.tenant_id AND id = NEW.scope_id
    ) THEN
        RAISE EXCEPTION 'agent compartment target does not exist';
    ELSIF NEW.scope_type = 'workspace' AND NOT EXISTS (
        SELECT 1 FROM public.workspaces
        WHERE tenant_id = NEW.tenant_id AND id = NEW.scope_id
    ) THEN
        RAISE EXCEPTION 'workspace compartment target does not exist';
    ELSIF NEW.scope_type = 'project' THEN
        SELECT workspace_id INTO project_workspace_id FROM public.projects
        WHERE tenant_id = NEW.tenant_id AND id = NEW.scope_id;
        IF NOT FOUND OR project_workspace_id <> parent_scope_id THEN
            RAISE EXCEPTION 'project compartment target does not belong to parent workspace';
        END IF;
    END IF;

    IF TG_OP = 'UPDATE' AND NEW.parent_id IS DISTINCT FROM OLD.parent_id AND
       parent_path <@ OLD.path THEN
        RAISE EXCEPTION 'compartment cycle';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER compartments_validate_hierarchy
BEFORE INSERT OR UPDATE OF parent_id, path, scope_type, scope_id ON compartments
FOR EACH ROW EXECUTE FUNCTION validate_compartment_hierarchy();

CREATE FUNCTION reject_document_cycle() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        WITH RECURSIVE ancestors(tenant_id, origin_id, id, parent_id) AS (
            SELECT n.tenant_id, n.id, d.id, d.parent_id
            FROM changed_documents n
            JOIN public.documents d
              ON d.tenant_id = n.tenant_id AND d.id = n.parent_id
            UNION
            SELECT a.tenant_id, a.origin_id, d.id, d.parent_id
            FROM ancestors a
            JOIN public.documents d
              ON d.tenant_id = a.tenant_id AND d.id = a.parent_id
        )
        SELECT 1 FROM ancestors WHERE id = origin_id
    ) THEN
        RAISE EXCEPTION 'document cycle';
    END IF;

    RETURN NULL;
END;
$$;

CREATE TRIGGER documents_reject_insert_cycle
AFTER INSERT ON documents
REFERENCING NEW TABLE AS changed_documents
FOR EACH STATEMENT EXECUTE FUNCTION reject_document_cycle();

CREATE TRIGGER documents_reject_update_cycle
AFTER UPDATE ON documents
REFERENCING NEW TABLE AS changed_documents
FOR EACH STATEMENT EXECUTE FUNCTION reject_document_cycle();

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenants FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenants
    USING (id = current_tenant_id())
    WITH CHECK (id = current_tenant_id());

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'tenant_memberships', 'agents', 'agent_delegations', 'agent_runs',
        'workspaces', 'workspace_memberships', 'projects', 'project_memberships',
        'mutation_idempotency', 'compartments', 'subsystems', 'project_subsystems', 'documents',
        'document_revisions', 'document_subsystems', 'chunks', 'document_links',
        'jobs', 'vector_index_state'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I USING (tenant_id = current_tenant_id()) WITH CHECK (tenant_id = current_tenant_id())',
            table_name
        );
    END LOOP;
END;
$$;

GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO memsystem_service;

GRANT UPDATE (name) ON tenants TO memsystem_service;
GRANT UPDATE (role, state) ON tenant_memberships TO memsystem_service;
GRANT UPDATE (name, archived_at) ON agents TO memsystem_service;
GRANT UPDATE (can_read, can_write, expires_at, revoked_at) ON agent_delegations TO memsystem_service;
GRANT UPDATE (state, expires_at) ON agent_runs TO memsystem_service;
GRANT UPDATE (name, version, archived_at) ON workspaces TO memsystem_service;
GRANT UPDATE (can_read, can_write) ON workspace_memberships TO memsystem_service;
GRANT UPDATE (name, root_hint, version, archived_at) ON projects TO memsystem_service;
GRANT UPDATE (can_read, can_write) ON project_memberships TO memsystem_service;
GRANT UPDATE (response) ON mutation_idempotency TO memsystem_service;
GRANT UPDATE (name) ON compartments TO memsystem_service;
GRANT UPDATE (name, description, version, archived_at) ON subsystems TO memsystem_service;
GRANT UPDATE (
    slug, current_revision_id, expires_at, deleted_at,
    auto_recall_approved_at, auto_recall_approved_by, updated_at
) ON documents TO memsystem_service;
GRANT UPDATE (
    embedding, embedding_model, embedding_version, embedding_dimension,
    embedding_state, embedding_error, embedding_attempted_at,
    indexed_revision, indexed_generation
) ON chunks TO memsystem_service;
GRANT UPDATE (weight) ON document_links TO memsystem_service;
GRANT UPDATE (state, attempts, error, next_attempt_at, claimed_at, updated_at)
    ON jobs TO memsystem_service;
GRANT UPDATE (generation, generation_path, checksum, job_high_water_mark, updated_at)
    ON vector_index_state TO memsystem_service;

GRANT DELETE ON document_links, project_subsystems, document_subsystems TO memsystem_service;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO memsystem_service;
GRANT EXECUTE ON FUNCTION current_tenant_id() TO memsystem_service;

RESET ROLE;
